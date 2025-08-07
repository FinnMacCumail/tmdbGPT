# TMDBGPT Development Workflow

## Overview
This document outlines the development workflow for TMDBGPT using a clean feature branch strategy with the `main` branch as the primary development base.

## Branch Strategy

### Local and Remote Branch Organization

#### **Primary Development Branch**
- **`main` (local and remote)**: Production-ready code with clean implementation
  - Contains essential debugging via 🧠 DEBUGGING SUMMARY REPORT
  - No temporary debug statements or development artifacts
  - Ready for deployment at any time
  - **This is where all active development happens**

#### **Historical Reference Branches**
- **`origin/reorg` (remote only)**: Historical debug-rich branch
  - Preserved on GitHub as reference for complex debugging scenarios  
  - Contains comprehensive debug logging and development artifacts
  - **NOT for active development** - use for reference only
  - Team members should not create new branches from this

- **`archive/reorg-debug-branch-2025-01-06` (local tag)**: Local archive snapshot
  - Git tag preserving the exact state when workflow changed
  - Available locally for historical analysis
  - Can be pushed to remote if needed: `git push origin --tags`

#### **Team Collaboration Guidelines**
- **All team members**: Work exclusively from `main` branch
- **New features**: Always branch from `main`, never from `origin/reorg`
- **Pull requests**: Always target `main` branch
- **Remote sync**: `git pull origin main` before starting new features

## Feature Development Workflow

### 1. Starting a New Feature
```bash
# Ensure you're on the latest main branch
git checkout main
git pull origin main

# Create a new feature branch
git checkout -b feature/description-of-feature

# Example naming conventions:
# feature/improved-search-algorithm
# feature/add-tv-series-support  
# feature/fix-constraint-parsing
```

### 2. Development Process
During development, you can add **temporary debug statements** as needed:

```python
# Temporary debugging during development - OK to add
print(f"🔧 Debug: Processing entity {entity_name}")
print(f"📊 Debug: Found {len(results)} results")
print(f"⚙️ Debug: Constraint tree state: {constraint_tree}")

# Use any debugging approach that helps you develop
logger.debug(f"Processing step {step_id}")
```

**Key Points:**
- Add debug statements freely during development
- Use descriptive emojis and prefixes for clarity
- Don't worry about production cleanliness during development

### 3. Pre-Merge Cleanup Checklist

Before merging your feature branch back to `main`, perform this cleanup:

#### ✅ Remove Temporary Debug Statements
- [ ] Remove all `print()` statements added during development
- [ ] Remove temporary debug logging calls
- [ ] Remove any debug imports that are no longer needed
- [ ] Replace with comments if the debug info might be useful for future maintenance

#### ✅ Preserve Production Debugging
- [ ] Ensure 🧠 DEBUGGING SUMMARY REPORT functionality remains intact
- [ ] Verify `log_summary()` calls are preserved where appropriate
- [ ] Keep ExecutionTraceLogger calls that provide production value

#### ✅ Code Quality
- [ ] Run any linting tools if available
- [ ] Ensure tests pass
- [ ] Remove any commented-out code blocks
- [ ] Clean up any temporary files

#### ✅ Example Cleanup
**Before cleanup (development code):**
```python
def process_entity(entity):
    print(f"🔧 Processing entity: {entity}")  # REMOVE
    result = complex_processing(entity)
    print(f"📊 Result: {result}")  # REMOVE  
    return result
```

**After cleanup (production code):**
```python
def process_entity(entity):
    # Process entity through complex algorithm
    result = complex_processing(entity)
    return result
```

### 4. Merging to Main
```bash
# After cleanup, merge to main
git checkout main
git merge feature/your-feature-name

# Push to remote
git push origin main

# Clean up feature branch
git branch -d feature/your-feature-name
```

## Debugging Strategy

### Production Debugging
The `main` branch includes sophisticated production debugging through:
- **🧠 DEBUGGING SUMMARY REPORT**: Comprehensive execution traces
- **ExecutionTraceLogger**: Step-by-step processing logs  
- **Constraint tree analysis**: Shows query processing logic
- **Entity resolution tracking**: Entity mapping and resolution

### Development Debugging
During feature development, use temporary debug statements:
- `print()` statements with clear prefixes
- Temporary logging calls
- Debug variables and intermediate results
- Any debugging approach that helps development

### Emergency Production Debugging
If production issues require deeper debugging:
1. Check 🧠 DEBUGGING SUMMARY REPORT output first
2. If more detail needed, create a temporary debug branch from `main`
3. Add specific debug statements for the issue
4. Investigate and fix
5. Apply the fix (without debug statements) back to `main`

## Benefits of This Workflow

1. **Clean Production Code**: `main` always contains professional, deployable code
2. **Development Freedom**: Add any debugging needed during feature development  
3. **Simple Merges**: No complex branch synchronization or merge conflicts
4. **Preserved Debugging**: Essential production debugging capabilities maintained
5. **Standard Git Workflow**: Any developer can understand and follow this process

## Best Practices

### Commit Messages
- Use clear, descriptive commit messages
- Prefix with feature scope when helpful
- Example: `feature/search: Add fuzzy matching for entity names`

### Testing
- Test your feature thoroughly before cleanup
- Ensure 🧠 DEBUGGING SUMMARY REPORT still works after your changes
- Verify no regressions in existing functionality

### Documentation
- Update relevant documentation if your feature changes user-facing behavior
- Add comments for complex business logic
- Update this workflow document if process changes

## Troubleshooting

### If You Accidentally Merge Debug Statements
```bash
# Create a cleanup commit
git checkout main
# Remove the debug statements
git add .
git commit -m "cleanup: Remove temporary debug statements"
git push origin main
```

### If You Need Historical Debug Information
```bash
# Option 1: View the local archived tag
git show archive/reorg-debug-branch-2025-01-06

# Option 2: Reference the remote historical branch
git fetch origin reorg
git log origin/reorg --oneline -10

# Option 3: Create temporary branch from local archive if needed
git checkout -b temp-debug-reference archive/reorg-debug-branch-2025-01-06

# Option 4: Create temporary branch from remote historical branch
git checkout -b temp-debug-reference origin/reorg
```

## Remote Branch Management

### Pushing Changes
```bash
# Standard workflow - push your feature work
git push origin main

# Push tags (including archive tags) to remote
git push origin --tags
```

### Team Synchronization
```bash
# Before starting any new feature
git checkout main
git pull origin main

# Never pull from origin/reorg - it's historical reference only
```

### Branch Status Overview
- **Active Development**: `main` (local) ↔ `origin/main` (remote)
- **Historical Reference**: `origin/reorg` (remote, read-only)
- **Local Archive**: `archive/reorg-debug-branch-2025-01-06` (tag)

## Questions or Issues
If you encounter any issues with this workflow or need clarification, refer to:
- The 🧠 DEBUGGING SUMMARY REPORT output for production debugging
- The `origin/reorg` branch or local archive tag for historical debugging examples
- This document for workflow clarification