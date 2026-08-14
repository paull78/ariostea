"""Fetch raw wikitext from the MediaWiki API.

The only networked part of the corpus build, kept to one function with an
injectable client so everything around it stays testable offline. Wikimedia
asks API clients to identify themselves, hence the explicit User-Agent.

Response shapes below were confirmed against the live API
(en.wikipedia.org), not just its documentation: a missing page, an invalid
title, a bad revid, and a rate-limit/block error object all come back
differently, and each is worth its own message rather than one generic
failure -- Task 8 fetches ~75 articles in one run, and a batch failure
(rate-limited, blocked) should read as one clear cause, not 75 identical
"no revision" errors that all look like typos in the manifest.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

USER_AGENT = "ariostea-eval-corpus/0.1 (https://github.com/paull78/ariostea; corpus build script)"


class WikiFetchError(RuntimeError):
    """An article could not be fetched, or came back in an unusable shape."""


@dataclass(frozen=True)
class FetchedArticle:
    """One article's fetched revision.

    `title` is whatever MediaWiki reports for the page actually returned --
    if `redirects=1` resolved a redirect, that is the redirect *target's*
    title (confirmed against the real API: requesting "USA" returns a page
    titled "United States"), not necessarily the title requested. Task 8
    compares this against the title it asked for to log when they diverge;
    the manifest title stays authoritative regardless of what's logged.
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
    `titles` lookup and, per the API's own documentation (confirmed live),
    a harmless no-op for `revids` -- it produces a warning field we never
    read, not an error, so a pinned-revid fetch still succeeds unaffected.
    """
    params: dict[str, str] = {
        "action": "query",
        "prop": "revisions",
        "rvprop": "content|ids",
        "rvslots": "main",
        "format": "json",
        # Pins the shape `["query"]["pages"]` is indexed as a *list* below.
        # formatversion=1 (the API's default) returns a dict keyed by
        # pageid instead, which would make that indexing wrong silently
        # rather than loudly -- so this is not optional.
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
    except httpx.HTTPError as exc:
        raise WikiFetchError(f"{lang}:{title}: request failed: {exc}") from exc
    if response.status_code >= 400:
        raise WikiFetchError(f"{lang}:{title}: HTTP {response.status_code} {response.text[:200]}")

    try:
        data = response.json()
    except ValueError as exc:  # ValueError covers JSONDecodeError
        raise WikiFetchError(
            f"{lang}:{title}: invalid JSON response: {response.text[:200]}"
        ) from exc

    # A rate-limited, blocked, or malformed request comes back as HTTP 200
    # with a top-level `error` object instead of `query` (confirmed live).
    # Worth naming explicitly rather than letting it fall through to the
    # generic "no revision" path below -- see the module docstring.
    if "error" in data:
        err = data["error"]
        raise WikiFetchError(
            f"{lang}:{title}: API error {err.get('code', '?')}: {err.get('info', data)}"
        )

    query = data.get("query", {})
    # A revid the API doesn't recognize comes back as
    # {"badrevids": {...}} with no "pages" key at all (confirmed live) --
    # distinct from a missing/invalid title, and worth naming as such.
    if "badrevids" in query:
        raise WikiFetchError(f"{lang}:{title}: badrevids: {query['badrevids']}")

    try:
        page = query["pages"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise WikiFetchError(
            f"{lang}:{title}: no revision in response: {response.text[:200]}"
        ) from exc

    if page.get("missing") or page.get("invalid"):
        # A missing page has no "invalidreason"; an invalid title does
        # (confirmed live) -- fall back to a generic label for the former.
        reason = page.get("invalidreason", "page is missing")
        raise WikiFetchError(f"{lang}:{title}: no revision ({reason})")

    try:
        revision = page["revisions"][0]
        main_slot = revision["slots"]["main"]
    except (KeyError, IndexError, TypeError) as exc:
        raise WikiFetchError(
            f"{lang}:{title}: no revision in response: {response.text[:200]}"
        ) from exc

    # Everything downstream (wikitext_to_markdown) assumes wikitext; a page
    # with any other content model would silently feed garbage into it.
    content_model = main_slot.get("contentmodel", "wikitext")
    if content_model != "wikitext":
        raise WikiFetchError(
            f"{lang}:{title}: page {page.get('title', title)!r} has content model "
            f"{content_model!r}, not wikitext"
        )

    try:
        return FetchedArticle(
            title=page["title"],
            revid=int(revision["revid"]),
            wikitext=main_slot["content"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise WikiFetchError(
            f"{lang}:{title}: no revision in response: {response.text[:200]}"
        ) from exc
