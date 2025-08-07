# Changelog

All notable changes to TMDBGPT will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation suite with installation, user guide, architecture, and API reference
- Contributing guidelines and troubleshooting documentation  
- Advanced usage guide with performance optimization techniques
- Professional README with quick start instructions

### Changed
- Reorganized documentation structure with dedicated `/docs` directory
- Moved DEVELOPMENT_WORKFLOW.md to `/docs` directory for better organization

## [2.1.0] - 2025-01-07

### Added
- User-friendly progress indicators for improved user experience
- DEBUG_MODE toggle to switch between user and developer modes
- Clean progress messages: "🔍 Understanding your question...", "🎭 Identifying people...", etc.
- Conditional debugging output control

### Changed  
- 🧠 DEBUGGING SUMMARY REPORT now only appears when DEBUG_MODE = True
- Default user experience is now clean and non-technical (DEBUG_MODE = False)
- Enhanced step_runner.py to respect DEBUG_MODE setting

### Fixed
- Silent processing periods replaced with informative progress indicators
- Improved user feedback during query processing

## [2.0.0] - 2025-01-06

### Added
- Complete branch reorganization with cleaned main branch
- Established feature branch workflow from main
- Archived historical debugging branch as reference
- Comprehensive branch management strategy

### Changed
- **BREAKING**: Replaced reorg branch workflow with main-based development
- Cleaned debug output except essential 🧠 DEBUGGING SUMMARY REPORT
- Streamlined codebase for production readiness

### Removed
- tmdb-gpt-ui frontend (archived)
- Temporary debug statements from production code
- Development artifacts and experimental code

### Infrastructure
- Recreated virtual environment with clean dependencies
- Updated git workflow documentation
- Established remote branch strategy with GitHub integration

## [1.x.x] - Historical Releases

### Background
Previous versions (1.x.x series) were developed in the `reorg` branch with extensive debugging and experimental features. These releases included:

- Multi-step query planner with semantic search
- ChromaDB vector storage for endpoint matching
- Symbolic constraint planning and validation
- Role-aware entity resolution
- Progressive fallback and constraint relaxation
- Comprehensive execution tracing and logging

The `reorg` branch is preserved as `origin/reorg` for historical reference and debugging scenarios, but is not used for active development.

---

## Release Notes Format

### Version Numbering
- **MAJOR**: Incompatible API changes or significant architectural changes
- **MINOR**: New functionality added in a backward-compatible manner  
- **PATCH**: Backward-compatible bug fixes

### Change Categories
- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Features removed in this release
- **Fixed**: Bug fixes
- **Security**: Security improvements

---

## Upcoming Features

### Planned for v2.2.0
- Enhanced query processing performance
- Extended entity type support
- Improved caching mechanisms
- Additional response formatting options

---

**Note**: This changelog will be updated with each release. For detailed technical changes, see the commit history and pull request discussions on GitHub.