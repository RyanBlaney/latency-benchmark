## One Interesting Observation 🤔:

The parser is detecting this as "VOD Media Playlist" but this appears to be a live stream based on:

Updating media sequence (5979045 → 5979049 between runs)
TuneIn headers like x-tunein-playlist-next-append
Short cache TTL (2-6 seconds)

The issue is that TuneIn's live streams don't include the standard #EXT-X-ENDLIST tag, so the parser defaults to is_live: false. This is actually correct behavior since many live streams work this way.

### Potential Enhancement:

**We could improve live detection by checking**:

```go
playlist.IsLive = !hasEndListTag(playlist) &&
(playlist.Headers["cache-control"] == "must-revalidate,stale-if-error=6" ||
playlist.Headers["x-tunein-playlist-next-append"] != "")
```
