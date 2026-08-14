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


def test_fetch_article_pins_by_revid_when_given_one():
    seen = {}

    def handler(request):
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json=_page())

    fetch_article(_client(handler), lang="en", title="Violin", revid=42)

    assert seen["params"]["revids"] == "42"
    assert "titles" not in seen["params"]


def test_fetch_article_raises_on_a_missing_page():
    def handler(request):
        return httpx.Response(200, json={"query": {"pages": [{"title": "Nope", "missing": True}]}})

    with pytest.raises(WikiFetchError, match="no revision"):
        fetch_article(_client(handler), lang="en", title="Nope")


def test_fetch_article_raises_on_an_http_error():
    def handler(request):
        return httpx.Response(503, text="down")

    with pytest.raises(WikiFetchError, match="503"):
        fetch_article(_client(handler), lang="en", title="Violin")


def test_fetch_article_sends_the_falsy_revid_as_a_query_param_not_the_title():
    """Regression test for a ternary bug in the plan's reference snippet:
    `params["revids" if revid is not None else "titles"] = str(revid) if
    revid else title` picks the key based on `revid is not None` but the
    value based on `revid`'s truthiness -- for `revid=0` those disagree, so
    the key would be "revids" but the value would be `title` (a string,
    not "0"). Real callers go through `ArticleSpec`, which already forbids
    revid 0, but this function has no such guard on its own signature, so
    it must still build the request correctly for any int it accepts.
    """
    seen = {}

    def handler(request):
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json=_page())

    fetch_article(_client(handler), lang="en", title="Violin", revid=0)

    assert seen["params"]["revids"] == "0"
    assert "titles" not in seen["params"]


def test_fetch_article_raises_a_clear_error_on_a_top_level_api_error():
    """A rate-limited or blocked client gets HTTP 200 with a top-level
    `error` object instead of `query` (confirmed against the real API this
    is how MediaWiki reports it, not an HTTP error status). This is the
    failure mode most likely to repeat identically across an entire batch
    of ~75 fetches, so it should surface as its own clear message rather
    than the generic "no revision in response"."""

    def handler(request):
        return httpx.Response(
            200, json={"error": {"code": "ratelimited", "info": "You've exceeded your rate limit."}}
        )

    with pytest.raises(WikiFetchError, match="ratelimited"):
        fetch_article(_client(handler), lang="en", title="Violin")


def test_fetch_article_raises_a_clear_error_on_bad_revids():
    """A revid the API doesn't recognize comes back as `{"query":
    {"badrevids": {...}}}` with no "pages" key at all (confirmed against the
    real API) -- distinct from the missing-page shape, and worth naming so
    it isn't confused with a bad title."""

    def handler(request):
        return httpx.Response(
            200,
            json={
                "query": {"badrevids": {"999999999999": {"revid": 999999999999, "missing": True}}}
            },
        )

    with pytest.raises(WikiFetchError, match="badrevids"):
        fetch_article(_client(handler), lang="en", title="Violin", revid=999999999999)


def test_fetch_article_raises_on_an_invalid_title_with_the_api_reason():
    """An invalid title (bad characters) comes back as `{"invalid": true,
    "invalidreason": "..."}` (confirmed against the real API) rather than
    `{"missing": true}`. The reason is worth surfacing since it names
    exactly what's wrong with the manifest entry."""

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

    with pytest.raises(WikiFetchError, match="invalid characters"):
        fetch_article(_client(handler), lang="en", title="Foo::Bar<>")


def test_fetch_article_returns_the_resolved_title_when_the_requested_title_redirects():
    """`redirects=1` resolves a redirect server-side and the returned page's
    title is the *target's* title, not the one requested (confirmed against
    the real API: requesting "USA" returns a page titled "United States").
    Task 8 compares the requested title against this to log when they
    differ, so the returned title must be what the API actually reports."""

    def handler(request):
        return httpx.Response(200, json=_page(title="United States"))

    article = fetch_article(_client(handler), lang="en", title="USA")

    assert article.title == "United States"


def test_fetch_article_raises_on_a_non_wikitext_content_model():
    """`slots.main.content` is assumed to be wikitext everywhere downstream
    (wikitext_to_markdown has no other input shape). A page whose content
    model is something else (e.g. a JSON or Scribunto page slipping into the
    manifest by mistake) should fail loudly here rather than feed
    non-wikitext into a wikitext parser."""

    def handler(request):
        page = _page()
        page["query"]["pages"][0]["revisions"][0]["slots"]["main"]["contentmodel"] = "json"
        return httpx.Response(200, json=page)

    with pytest.raises(WikiFetchError, match="json"):
        fetch_article(_client(handler), lang="en", title="Violin")
