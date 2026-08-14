"""Fetch raw wikitext from the MediaWiki API.

The only networked part of the corpus build, kept to one function with an
injectable client so everything around it stays testable offline. Wikimedia
asks API clients to identify themselves, hence the explicit User-Agent.

Response shapes below were confirmed against the live API
(en.wikipedia.org), not just its documentation: a missing page, an invalid
title, a bad revid, and a rate-limit/block error object all come back
differently, and each gets its own message rather than one generic
failure -- a corpus build fetches ~75 articles in one run, and a failure
that repeats identically across the whole batch (rate-limited, blocked)
should read as one clear cause, not 75 messages that all look like typos in
the manifest.

Every branch below is deliberately given wording that cannot appear in an
echoed `response.text[:200]` snippet from the mock bodies its test uses (a
fixed English phrase this module wrote, not a JSON key or a value pulled out
of the response). That is what makes each branch's test fail if the branch
is deleted, rather than silently passing because the *fallback* message
happened to echo the same substring back out of the raw JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

USER_AGENT = "ariostea-eval-corpus/0.1 (https://github.com/paull78/ariostea; corpus build script)"

# HTTP statuses worth retrying after a delay. Deliberately narrow: this is
# the exact set called out as transient, not "everything >= 500" -- a 5xx
# outside this set is left non-retryable pending a broader policy decision
# elsewhere, rather than guessed at here.
_RETRYABLE_STATUSES = frozenset({429, 503})


class WikiFetchError(RuntimeError):
    """An article could not be fetched, or came back in an unusable shape.

    `retryable` tells a caller whether trying again after a delay is worth
    it, or whether nothing will change: True for failure modes that are
    inherently transient (a 429/503 status, a transport-level connection
    failure, or the API's own rate-limit/block error object) and False for
    failure modes that describe the request itself (a bad title, a bad
    revid, a page whose content model isn't wikitext, a malformed body) --
    retrying an unrecognized revid a second time gets the same `badrevids`
    response every time. This function is the only layer that has seen the
    raw response well enough to make that call; a caller working from the
    message string alone cannot reliably re-derive it.
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class FetchedArticle:
    """One article's fetched revision.

    `title` is whatever MediaWiki reports for the page actually returned --
    if `redirects=1` resolved a redirect, that is the redirect *target's*
    title (confirmed against the real API: requesting "USA" returns a page
    titled "United States"), not necessarily the title requested. A caller
    that wants to detect that divergence (log it, say) compares this
    against the title it asked for.
    """

    title: str
    revid: int
    wikitext: str


def fetch_article(
    client: httpx.Client, lang: str, title: str, revid: int | None = None
) -> FetchedArticle:
    """Return one article's wikitext: the current revision by `title`, or an
    exact pinned revision by `revid` (reproducing a committed snapshot byte
    for byte).

    Passing `revid` switches the request from a `titles` lookup to a
    `revids` lookup entirely -- MediaWiki does not accept both at once.
    `redirects=1` is still sent unconditionally: it is meaningful for a
    `titles` lookup and, per the API's own documentation (confirmed live), a
    harmless no-op for `revids` -- it produces a warning field this function
    never reads, not an error, so a pinned-revid fetch still succeeds
    unaffected.

    `lang` is not validated here -- it is interpolated directly into the
    request host (`{lang}.wikipedia.org`). The plan this function was built
    for has every caller construct `lang` via `ArticleSpec`, which restricts
    it to `^[a-z]{2,3}$` before it would ever reach here, so validating it
    again here would be redundant defense; but that caller doesn't exist in
    this codebase yet; this module's own tests pass `lang` as a bare string
    literal. A `lang` that breaks URL construction outright (a newline, a
    control character) still fails safely here -- see the `InvalidURL`
    handling below -- it just fails with a transport-level message instead
    of a validation one.

    Response bytes are parsed as JSON straight off `response.content`
    (`httpx.Response.json()` is `json.loads` over raw bytes, strict UTF-8);
    `Content-Type`'s declared charset is never consulted for that path, only
    for `response.text`, which this function uses solely inside error
    messages. Article wikitext this size (well under a megabyte, even for
    it/es articles heavy in accented text) is far short of where the API's
    own result-size limits would truncate a single-page response, so that
    failure mode isn't guarded against here.
    """
    params: dict[str, str] = {
        "action": "query",
        "prop": "revisions",
        "rvprop": "content|ids",
        "rvslots": "main",
        "format": "json",
        # Pins the shape `["query"]["pages"]` is indexed as a *list* below.
        # formatversion=1 (the API's default) would make `pages` a dict
        # keyed by pageid instead, so `pages[0]` would raise KeyError (there
        # is no key `0`) -- not silently wrong, but not obviously right
        # either: it would be caught below and reported as "no pages in the
        # API response", a true statement about the mis-indexed access that
        # gives no hint the real cause is a missing formatversion param.
        "formatversion": "2",
        "redirects": "1",
    }
    if revid is not None:
        params["revids"] = str(revid)
    else:
        params["titles"] = title

    try:
        response = client.get(
            f"https://{lang}.wikipedia.org/w/api.php",
            params=params,
            headers={"User-Agent": USER_AGENT},
        )
    except httpx.InvalidURL as exc:
        # Not a subclass of HTTPError -- raised while building the request,
        # before any network call, typically because `lang` or `title`
        # contains a character (newline, control character) that can't
        # appear in a URL at all. Not transient: the same bad input
        # produces the same failure every time.
        raise WikiFetchError(f"{lang}:{title}: request URL was invalid: {exc}") from exc
    except httpx.HTTPError as exc:
        # Connection failures, timeouts, and the like -- worth another try
        # once whatever's interrupting the network clears up.
        raise WikiFetchError(f"{lang}:{title}: request failed: {exc}", retryable=True) from exc

    if not response.is_success:
        # `response.is_success` (2xx only) rather than `status_code >= 400`:
        # this client may not follow redirects (the injected-client design
        # means that's the caller's choice, not this function's), and a 3xx
        # response has no JSON body to parse -- treating it as success would
        # otherwise fall through to a confusing "response was not valid
        # JSON" error that never mentions the redirect that caused it.
        raise WikiFetchError(
            f"{lang}:{title}: HTTP {response.status_code}: {response.text[:200]}",
            retryable=response.status_code in _RETRYABLE_STATUSES,
        )

    try:
        data: Any = response.json()
    except ValueError as exc:  # ValueError covers JSONDecodeError
        raise WikiFetchError(
            f"{lang}:{title}: response was not valid JSON: {response.text[:200]}"
        ) from exc

    if not isinstance(data, dict):
        raise WikiFetchError(
            f"{lang}:{title}: API response was not a JSON object "
            f"(got {type(data).__name__}): {response.text[:200]}"
        )

    # A rate-limited, blocked, or malformed request comes back as HTTP 200
    # with a top-level `error` object instead of `query` (confirmed live) --
    # the exact scenario `retryable` exists for, so it is marked transient.
    if "error" in data:
        err = data["error"]
        if not isinstance(err, dict):
            raise WikiFetchError(
                f"{lang}:{title}: MediaWiki reported a malformed error object: {err!r}"
            )
        code = err.get("code", "?")
        info = err.get("info", "")
        raise WikiFetchError(
            f"{lang}:{title}: MediaWiki API returned an error (code={code}): {info}",
            retryable=True,
        )

    query = data.get("query")
    if not isinstance(query, dict):
        raise WikiFetchError(
            f"{lang}:{title}: API response had no usable query result: {response.text[:200]}"
        )

    # A revid the API doesn't recognize comes back as {"badrevids": {...}}
    # with no "pages" key at all (confirmed live) -- distinct from a
    # missing/invalid title, and not transient: the same revid is rejected
    # every time.
    if "badrevids" in query:
        raise WikiFetchError(
            f"{lang}:{title}: revid was rejected by the API as unrecognized "
            f"(badrevids): {query['badrevids']}"
        )

    pages = query.get("pages")
    if not isinstance(pages, list) or not pages:
        raise WikiFetchError(
            f"{lang}:{title}: API response contained no pages: {response.text[:200]}"
        )
    page = pages[0]
    if not isinstance(page, dict):
        raise WikiFetchError(
            f"{lang}:{title}: API response's page entry was not an object: {response.text[:200]}"
        )

    if page.get("missing") or page.get("invalid"):
        # A missing page has no "invalidreason"; an invalid title does
        # (confirmed live) -- fall back to a generic label for the former.
        reason = page.get("invalidreason", "the page does not exist")
        raise WikiFetchError(f"{lang}:{title}: the requested page is unavailable ({reason})")

    revisions = page.get("revisions")
    if not isinstance(revisions, list) or not revisions:
        raise WikiFetchError(
            f"{lang}:{title}: page has no revisions in the API response: {response.text[:200]}"
        )
    revision = revisions[0]

    try:
        main_slot = revision["slots"]["main"]
    except (KeyError, TypeError) as exc:
        raise WikiFetchError(
            f"{lang}:{title}: revision has no usable content slot: {response.text[:200]}"
        ) from exc

    # Everything downstream (wikitext_to_markdown) assumes wikitext; a page
    # with any other content model would silently feed garbage into it.
    content_model = (
        main_slot.get("contentmodel", "wikitext") if isinstance(main_slot, dict) else None
    )
    if content_model != "wikitext":
        raise WikiFetchError(
            f"{lang}:{title}: page {page.get('title', title)!r} has unsupported content model "
            f"{content_model!r} (expected wikitext)"
        )

    try:
        return FetchedArticle(
            title=page["title"],
            revid=int(revision["revid"]),
            wikitext=main_slot["content"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise WikiFetchError(
            f"{lang}:{title}: revision was missing an expected field: {response.text[:200]}"
        ) from exc
