#!/usr/bin/env python3

# Debug the director detection logic

query_text = "who directed inception?"

# Check if director question is detected
is_director_question = any(keyword in query_text for keyword in ["direct", "director", "directed"])

print(f"Query: {query_text}")
print(f"Is director question detected: {is_director_question}")
print(f"Keywords checked: ['direct', 'director', 'directed']")

# Check individual keywords
for keyword in ["direct", "director", "directed"]:
    found = keyword in query_text
    print(f"  '{keyword}' in query: {found}")