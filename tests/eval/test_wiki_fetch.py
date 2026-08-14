import httpx
import pytest

from ariostea.eval.wiki_fetch import FetchedArticle, WikiFetchError, fetch_article


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def _page(**over):
    page = {"title": "Violin", "revisions": [{"revid": 42, "slots": {"main": {"content": "raw"}}}]}
    page.update(over)
    return {"query": {"pages": [page]}}


def test_fetch_article_requests_wikitext_by_title_and_returns_the_revision():
    seen = {}

    def handler(request):
        seen["host"] = request.url.host
        seen["params"] = dict(request.url.params)
        seen["agent"] = request.headers.get("user-agent")
        return httpx.Response(200, json=_page())

    article = fetch_article(_client(handler), lang="en", title="Violin")

    assert article == FetchedArticle(title="Violin", revid=42, wikitext="raw")
    assert seen["host"] == "en.wikipedia.org"
    assert seen["params"]["titles"] == "Violin"
    assert seen["params"]["rvslots"] == "main"
    assert seen["params"]["redirects"] == "1"
    assert "ariostea" in seen["agent"]


def test_fetch_article_requests_content_and_ids_in_formatversion_2():
    """Pins two request-shaping params a passing suite previously didn't
    check at all: `formatversion=2` (required for the `pages[0]` list
    indexing below to be correct rather than an accidental KeyError) and
    `rvprop=content|ids` (without "ids" there is no revid to report back).
    Deleting either line from the request params now fails here directly,
    rather than only being caught (misleadingly) by a downstream shape
    error."""
    seen = {}

    def handler(request):
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json=_page())

    fetch_article(_client(handler), lang="en", title="Violin")

    assert seen["params"]["formatversion"] == "2"
    assert seen["params"]["rvprop"] == "content|ids"


def test_fetch_article_pins_by_revid_when_given_one():
    seen = {}

    def handler(request):
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json=_page())

    fetch_article(_client(handler), lang="en", title="Violin", revid=42)

    assert seen["params"]["revids"] == "42"
    assert "titles" not in seen["params"]


def test_fetch_article_sends_the_falsy_revid_as_a_query_param_not_the_title():
    """Regression test for a ternary bug in an earlier draft:
    `params["revids" if revid is not None else "titles"] = str(revid) if
    revid else title` picks the key based on `revid is not None` but the
    value based on `revid`'s truthiness -- for `revid=0` those disagree, so
    the key would be "revids" but the value would be `title` (a string,
    not "0"). Real callers are expected to go through `ArticleSpec`, which
    already forbids revid 0, but this function has no such guard on its own
    signature, so it must still build the request correctly for any int it
    accepts.
    """
    seen = {}

    def handler(request):
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json=_page())

    fetch_article(_client(handler), lang="en", title="Violin", revid=0)

    assert seen["params"]["revids"] == "0"
    assert "titles" not in seen["params"]


def test_fetch_article_raises_on_an_http_error_and_marks_it_retryable():
    def handler(request):
        return httpx.Response(503, text="down")

    with pytest.raises(WikiFetchError, match="HTTP 503") as exc_info:
        fetch_article(_client(handler), lang="en", title="Violin")
    assert exc_info.value.retryable is True


def test_fetch_article_marks_a_429_retryable_too():
    def handler(request):
        return httpx.Response(429, text="slow down")

    with pytest.raises(WikiFetchError, match="HTTP 429") as exc_info:
        fetch_article(_client(handler), lang="en", title="Violin")
    assert exc_info.value.retryable is True


def test_fetch_article_does_not_mark_a_404_retryable():
    """Deliberately narrow: only 429/503 are treated as transient. A 404
    (or any other status outside that set) is left non-retryable rather than
    guessed at -- retrying it would get the same response every time."""

    def handler(request):
        return httpx.Response(404, text="not found")

    with pytest.raises(WikiFetchError, match="HTTP 404") as exc_info:
        fetch_article(_client(handler), lang="en", title="Violin")
    assert exc_info.value.retryable is False


def test_fetch_article_treats_a_redirect_status_as_an_error_not_success():
    """`is_success` (2xx only), not `status_code >= 400`: this function's
    client may not follow redirects (the test client here defaults to not
    following them, same as the library default), and a bare 3xx has no
    JSON body. Before this used `is_success`, a 3xx fell through to the JSON
    parser and produced a confusing "response was not valid JSON" error
    that never mentioned the redirect."""

    def handler(request):
        return httpx.Response(302, headers={"location": "https://en.wikipedia.org/wiki/Violin"})

    with pytest.raises(WikiFetchError, match="HTTP 302") as exc_info:
        fetch_article(_client(handler), lang="en", title="Violin")
    assert "not valid JSON" not in str(exc_info.value)


def test_fetch_article_raises_on_a_transport_error_and_marks_it_retryable():
    def handler(request):
        raise httpx.ConnectTimeout("timed out")

    with pytest.raises(WikiFetchError, match="request failed") as exc_info:
        fetch_article(_client(handler), lang="en", title="Violin")
    assert exc_info.value.retryable is True


def test_fetch_article_raises_on_an_invalid_request_url_and_does_not_mark_it_retryable():
    """`httpx.InvalidURL` is not an `HTTPError` subclass, so it needs its
    own `except` clause -- raised while building the request (before any
    network call) when a component can't appear in a URL at all, confirmed
    live: a `lang` containing a bare newline triggers it. Not transient:
    the same bad input fails the same way every time, so unlike a
    connection failure this should not be retried blindly."""

    def handler(request):
        raise AssertionError("must not reach the transport")

    with pytest.raises(WikiFetchError, match="invalid") as exc_info:
        fetch_article(_client(handler), lang="en\n", title="Violin")
    assert exc_info.value.retryable is False


def test_fetch_article_raises_on_malformed_json_body():
    def handler(request):
        return httpx.Response(200, content=b"this is not json")

    with pytest.raises(WikiFetchError, match="not valid JSON") as exc_info:
        fetch_article(_client(handler), lang="en", title="Violin")
    assert exc_info.value.retryable is False


@pytest.mark.parametrize("raw", [b"null", b"42", b"[1, 2]", b'"hello"'])
def test_fetch_article_raises_cleanly_on_a_json_body_that_is_not_an_object(raw):
    """A response that is valid JSON but not a `{...}` object at the top
    level (`null`, a bare number, a list, a bare string) used to escape as
    a `TypeError`/`AttributeError` from the first `dict`-shaped access
    (`"error" in data`, `data.get(...)`) -- breaking the "every failure is a
    WikiFetchError" contract callers rely on to catch just one exception
    type. All four must now raise WikiFetchError instead.

    Built from raw JSON bytes rather than `httpx.Response(..., json=body)`:
    passing `json=None` specifically makes httpx skip setting a body at all
    (indistinguishable from an empty response), not serialize the literal
    `null` this case needs."""

    def handler(request):
        return httpx.Response(200, content=raw, headers={"content-type": "application/json"})

    with pytest.raises(WikiFetchError, match="not a JSON object") as exc_info:
        fetch_article(_client(handler), lang="en", title="Violin")
    assert exc_info.value.retryable is False


def test_fetch_article_raises_cleanly_on_a_malformed_error_object():
    """`{"error": "some string"}` used to escape as an `AttributeError` from
    `err.get(...)` assuming `error` is always a dict. MediaWiki's own error
    objects always are, in practice, but a malformed one should not crash
    the batch either."""

    def handler(request):
        return httpx.Response(200, json={"error": "boom"})

    with pytest.raises(WikiFetchError, match="malformed error object") as exc_info:
        fetch_article(_client(handler), lang="en", title="Violin")
    assert exc_info.value.retryable is False


def test_fetch_article_raises_a_clear_retryable_error_on_a_top_level_api_error():
    """A rate-limited or blocked client gets HTTP 200 with a top-level
    `error` object instead of `query` (confirmed against the real API this
    is how MediaWiki reports it, not an HTTP error status). Marked
    retryable=True: this is the exact scenario -- Wikimedia telling the
    client to back off -- that `retryable` exists to signal.

    The match text ("MediaWiki API returned an error") is this module's own
    fixed phrase, not anything present verbatim in the mock's JSON body --
    unlike matching on "ratelimited" (which also appears in the raw
    response text and so would still match even if this whole branch were
    deleted and the generic "no usable query result" fallback fired
    instead, echoing the same raw body back in its message).
    """

    def handler(request):
        return httpx.Response(
            200, json={"error": {"code": "ratelimited", "info": "You've exceeded your rate limit."}}
        )

    with pytest.raises(WikiFetchError, match="MediaWiki API returned an error") as exc_info:
        fetch_article(_client(handler), lang="en", title="Violin")
    assert exc_info.value.retryable is True


def test_fetch_article_raises_on_a_response_with_no_query_or_error():
    """A body that is a JSON object but has neither "error" nor a "query"
    dict (e.g. just `{"batchcomplete": true}`) must still raise cleanly
    rather than blowing up on `"badrevids" in query` with `query=None`."""

    def handler(request):
        return httpx.Response(200, json={"batchcomplete": True})

    with pytest.raises(WikiFetchError, match="no usable query result") as exc_info:
        fetch_article(_client(handler), lang="en", title="Violin")
    assert exc_info.value.retryable is False


def test_fetch_article_raises_a_clear_error_on_bad_revids():
    """A revid the API doesn't recognize comes back as `{"query":
    {"badrevids": {...}}}` with no "pages" key at all (confirmed against the
    real API) -- distinct from the missing-page shape.

    Matches on "rejected by the API", this module's own fixed phrase, not
    on "badrevids" itself -- "badrevids" is a literal JSON key in the mock
    body, so it would still appear in the generic fallback's echoed
    `response.text[:200]` even if this branch were deleted entirely.
    """

    def handler(request):
        return httpx.Response(
            200,
            json={
                "query": {"badrevids": {"999999999999": {"revid": 999999999999, "missing": True}}}
            },
        )

    with pytest.raises(WikiFetchError, match="rejected by the API") as exc_info:
        fetch_article(_client(handler), lang="en", title="Violin", revid=999999999999)
    assert exc_info.value.retryable is False


def test_fetch_article_raises_on_a_missing_page():
    """Matches on "requested page is unavailable", this module's own fixed
    phrase -- not on "missing", which is also a literal JSON key in the
    mock body and so would leak through the generic fallback's echoed
    response text even with this branch deleted."""

    def handler(request):
        return httpx.Response(200, json={"query": {"pages": [{"title": "Nope", "missing": True}]}})

    with pytest.raises(WikiFetchError, match="requested page is unavailable") as exc_info:
        fetch_article(_client(handler), lang="en", title="Nope")
    assert exc_info.value.retryable is False


def test_fetch_article_raises_on_an_invalid_title_with_the_api_reason():
    """An invalid title (bad characters) comes back as `{"invalid": true,
    "invalidreason": "..."}` (confirmed against the real API) rather than
    `{"missing": true}`. Matches on "requested page is unavailable" (this
    module's own fixed phrase, shared with the missing-page branch above --
    both describe "no revision for this title") plus the API's own reason
    text, which *is* legitimately worth echoing here since it names exactly
    what's wrong with the manifest entry."""

    def handler(request):
        return httpx.Response(
            200,
            json={
                "query": {
                    "pages": [
                        {
                            "title": "Foo::Bar<>",
                            "invalid": True,
                            "invalidreason": 'The requested page title contains invalid characters: "<".',
                        }
                    ]
                }
            },
        )

    with pytest.raises(WikiFetchError, match="requested page is unavailable") as exc_info:
        fetch_article(_client(handler), lang="en", title="Foo::Bar<>")
    assert "invalid characters" in str(exc_info.value)


def test_fetch_article_returns_the_resolved_title_when_the_requested_title_redirects():
    """`redirects=1` resolves a redirect server-side and the returned page's
    title is the *target's* title, not the one requested (confirmed against
    the real API: requesting "USA" returns a page titled "United States").
    A caller comparing the requested title against this can detect and log
    the divergence; the returned title must be what the API actually
    reports for that to work."""

    def handler(request):
        return httpx.Response(200, json=_page(title="United States"))

    article = fetch_article(_client(handler), lang="en", title="USA")

    assert article.title == "United States"


def test_fetch_article_raises_on_an_empty_pages_list():
    def handler(request):
        return httpx.Response(200, json={"query": {"pages": []}})

    with pytest.raises(WikiFetchError, match="contained no pages") as exc_info:
        fetch_article(_client(handler), lang="en", title="Violin")
    assert exc_info.value.retryable is False


def test_fetch_article_raises_on_a_page_with_no_revisions():
    def handler(request):
        return httpx.Response(200, json={"query": {"pages": [{"title": "Violin"}]}})

    with pytest.raises(WikiFetchError, match="page has no revisions") as exc_info:
        fetch_article(_client(handler), lang="en", title="Violin")
    assert exc_info.value.retryable is False


def test_fetch_article_raises_on_a_non_wikitext_content_model():
    """`slots.main.content` is assumed to be wikitext everywhere downstream
    (wikitext_to_markdown has no other input shape). A page whose content
    model is something else (e.g. a JSON or Scribunto page slipping into the
    manifest by mistake) should fail loudly here rather than feed
    non-wikitext into a wikitext parser.

    Matches on "unsupported content model", this module's own fixed phrase
    -- not on the bare value "json", which also appears as a literal JSON
    value in the mock body (`"contentmodel": "json"`) and so would leak
    through the generic fallback's echoed response text even with this
    branch deleted."""

    def handler(request):
        page = _page()
        page["query"]["pages"][0]["revisions"][0]["slots"]["main"]["contentmodel"] = "json"
        return httpx.Response(200, json=page)

    with pytest.raises(WikiFetchError, match="unsupported content model") as exc_info:
        fetch_article(_client(handler), lang="en", title="Violin")
    assert exc_info.value.retryable is False


def test_fetch_article_raises_when_the_page_is_missing_a_title():
    """Exercises the final construction guard: pages/revisions/slots are
    all present and well-shaped, but the page object itself is missing
    "title". Confirmed (per review) that `int(revision["revid"])` cannot
    escape this same try block under formatversion=2, since revid always
    arrives as a JSON number there, not a string -- so this guard's other
    job is catching a structurally sound but incomplete final page object,
    not a revid coercion failure."""

    def handler(request):
        return httpx.Response(
            200,
            json={
                "query": {
                    "pages": [{"revisions": [{"revid": 42, "slots": {"main": {"content": "raw"}}}]}]
                }
            },
        )

    with pytest.raises(WikiFetchError, match="missing an expected field") as exc_info:
        fetch_article(_client(handler), lang="en", title="Violin")
    assert exc_info.value.retryable is False


def test_fetch_article_raises_when_revid_cannot_be_coerced_to_int():
    def handler(request):
        return httpx.Response(
            200,
            json=_page(
                revisions=[{"revid": "not-a-number", "slots": {"main": {"content": "raw"}}}]
            ),
        )

    with pytest.raises(WikiFetchError, match="missing an expected field"):
        fetch_article(_client(handler), lang="en", title="Violin")
