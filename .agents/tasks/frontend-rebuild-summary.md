# Frontend Rebuild Summary

## Date
Applied as part of workflow step `wf_b73106a1470c9ef7`.

## File Changed

### `index.html` (complete rewrite — 1500 lines)

A fully self-contained single-file frontend was written from scratch replacing the previous ~300-line version.

---

## What Was Built

### Visual Theme
- Dark navy background (`#0f172a`), card surfaces (`#1e293b`), border (`#334155`)
- Accent blue (`#3b82f6`) and indigo (`#6366f1`)
- Inter font loaded from Google Fonts
- CSS custom properties for all colors; all animations via `@keyframes`

### Header
- 🤖 **CareerAgent RAG** logo with animated gradient underline (`gradientShift` keyframe)
- Tagline: "AI-Powered Job Matching with Semantic Search"

### System Status Bar (`id="statusBar"`)
- 3 status pills: ChromaDB doc count, embedding model, Groq LLM
- Skeleton pulse animation while loading; red "⚠️ API Offline" on failure
- Populated by `GET /vector-db/status` on page load

### Main Layout
- CSS Grid — 2 columns on desktop (`1fr 1fr`), stacked on mobile (≤900px)

### Left Column
- **Tab switcher** — "📄 Paste Resume" / "📁 Upload File" with active blue underline
- **Paste tab**: `<textarea id="resumeTextarea">` (min-height 280px) with realistic placeholder resume; live character counter (`id="charCount"`) — red < 50 chars, green ≥ 50
- **Upload tab**: drag-and-drop zone (`id="uploadZone"`) with dragenter/dragover/dragleave/drop handlers; "Browse Files" button opens `<input type="file" id="fileInput" accept=".pdf,.docx,.txt">`; file info row shows name, size (KB/MB), ✅ icon and × clear button
- **How It Works pipeline**: 5 steps (Resume → Embeddings → Vector Search → LLM Reasoning → Matches) as cards with arrows; stacks vertically on mobile
- **Scrape panel**: `<details>` element (closed by default) with query, location, num_results, backend inputs and "🔍 Scrape & Index" button calling `POST /scrape-jobs`
- **Find My Job Matches → button** (`id="matchBtn"`) with gradient, loading state, disabled state
- **Stats pills** (`id="statsContainer"`) — hidden until results arrive; shows match count and top score

### Right Column
- **Empty state** (`id="emptyState"`) — inline SVG (document + magnifying glass), pulse animation
- **Results container** (`id="resultsContainer"`) — match cards injected here; each card has:
  - Bold title + company (muted)
  - Score badge (green ≥70, yellow 40–69, red <40)
  - Progress bar filled to match_score %
  - Matched skills (green pills `#166534`/`#22c55e`)
  - Missing skills (orange pills `#92400e`/`#fb923c`)
  - AI Reasoning with 3-line CSS clamp + "Show more" toggle
  - Staggered slide-in animation (`index * 100ms` delay)
- **Vector DB panel** (`id="vectorDbPanel"`) — always visible; shows collection, documents, embeddings, path, status dot; ↻ refresh button re-fetches `/vector-db/status`

### Toast Notifications
- `showToast(message, type)` — fixed top-right, slides in from right, auto-dismisses after 4s
- Green left border for success, red for error; multiple toasts stack

### Footer
- Tech stack credits + "API Docs →" link to `/docs`

---

## JavaScript Functions

| Function | Description |
|---|---|
| `loadVectorDBStatus()` | `GET /vector-db/status` → populates status bar + VDB panel |
| `matchJobs()` | Dispatches to `/match-jobs` (paste) or `/upload-resume` (file) |
| `scrapeJobs()` | `POST /scrape-jobs` → shows result, refreshes VDB status |
| `renderMatches(matches)` | Builds and animates match cards |
| `buildMatchCard(match, i)` | Constructs a single match card DOM element |
| `toggleReason(i)` | Toggles 3-line clamp on AI reasoning text |
| `setSelectedFile(file)` | Validates extension, stores file, shows file info |
| `clearFile()` | Resets upload state |
| `showToast(msg, type)` | Renders a stacking toast notification |
| `switchTab(name)` | Switches between paste/upload tabs |

## Verification
- **Line count**: 1500 (spec requires ≥ 400) ✅
- **Patterns found**: `upload-resume`, `uploadZone`, `vectorDbPanel`, `emptyState` — all present ✅
