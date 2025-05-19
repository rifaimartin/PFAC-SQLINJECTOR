

import random
import csv

# Collect all unique patterns
raw_patterns = [
    "' or", "\" or", "' ||", "\" ||", "= or", "= ||",
    "' =", "' >=", "' <=", "' <>", "\" =", "\" !=", "= =", "= <", " >=", " <=",
    "' union", "' select", "' from", "union select", "select from",
    "' convert(", "' avg(", "' round(", "' sum(", "' max(", "' min(",
    ") convert(", ") avg(", ") round(", ") sum(", ") max(", ") min(",
    "' delete", "' drop", "' insert", "' truncate", "' update", "' alter",
    ", delete", "; drop", "; insert", "; delete", ", drop", "; truncate", "; exec",
    "xp_cmdshell", "; truncate", "' ; update",
    "like or", "like ||", "' %", "like %", " %",
    "</script>", "</script >",
    "union", "select", "drop", "insert", "delete", "update",
    "or 1=1", "--", "#", "/*", "*/",
    "sleep(", "benchmark(", "count(*)",
    "information_schema.schemata", "null", "version(", "current_user", "outfile", "load_file",
    "union select", "union all select", "union select null",
    "information_schema", "information_schema.schemata", "from information_schema",
    "version()", "current_user", "database()", "schema_name",
    "select sleep", "case when", "then sleep",
    "select count(*)", "select count(*) from users",
    "union select null version", "union select null database", "union select null current_user"
]

# Remove duplicates and sort
unique_patterns = sorted(list(set(raw_patterns)))
print(f"Total unique patterns: {len(unique_patterns)}")

# Define risk categories and their corresponding scores
risk_categories = {
    "low": 10,
    "medium": 40,
    "high": 70,
    "critical": 100
}

# Define risk for each pattern
pattern_risks = {}

# Critical patterns (highest risk)
critical_patterns = [
    "xp_cmdshell", "; exec", "outfile", "load_file", "</script>", "</script >",
    "; drop", "' drop", "drop table", "drop database", 
    "; truncate", "' truncate", "truncate table"
]

# High risk patterns
high_patterns = [
    "union select", "information_schema", "or 1=1", "' or", "\" or",
    "from information_schema", "/*", "*/", "sleep(", "benchmark(",
    "select sleep", "case when", "then sleep", "' delete", "; delete",
    "version()", "database()", "current_user"
]

# Medium risk patterns
medium_patterns = [
    "; insert", "; update", "version(", "schema_name", "count(*)",
    "' select", "' union", "--", "#", "' alter", "' update", 
    "select count(*)", "select count(*) from users", "union all select"
]

# Assign risk to each pattern
for pattern in unique_patterns:
    # Check if any critical pattern is in this pattern
    if any(cp in pattern for cp in critical_patterns):
        pattern_risks[pattern] = "critical"
    # Check if any high risk pattern is in this pattern
    elif any(hp in pattern for hp in high_patterns):
        pattern_risks[pattern] = "high"
    # Check if any medium risk pattern is in this pattern
    elif any(mp in pattern for mp in medium_patterns):
        pattern_risks[pattern] = "medium"
    # Default to low risk
    else:
        pattern_risks[pattern] = "low"

# Generate 512 patterns with their risk levels and scores
num_patterns = 512
patterns_data = []

# Ensure we get some of each risk level by creating a balanced set first
balanced_patterns = []

for risk in risk_categories.keys():
    # Filter patterns by risk
    risk_patterns = [p for p in unique_patterns if pattern_risks[p] == risk]
    # If we have patterns of this risk, select some
    if risk_patterns:
        # Take about 1/4 of our total for each risk category (or fewer if not enough exist)
        num_to_select = min(len(risk_patterns), num_patterns // 4)
        selected = random.sample(risk_patterns, num_to_select)
        for pattern in selected:
            balanced_patterns.append((pattern, risk, risk_categories[risk]))

# Fill the rest randomly
remaining = num_patterns - len(balanced_patterns)
if remaining > 0:
    # Generate more patterns by randomly selecting from unique_patterns
    random_selections = random.choices(unique_patterns, k=remaining)
    for pattern in random_selections:
        risk = pattern_risks[pattern]
        patterns_data.append((pattern, risk, risk_categories[risk]))

# Combine balanced and random patterns
patterns_data = balanced_patterns + patterns_data

# Shuffle the patterns
random.shuffle(patterns_data)

# Just to make sure we have exactly 512
patterns_data = patterns_data[:num_patterns]

# Count patterns by risk
risk_counts = {risk: 0 for risk in risk_categories}
for _, risk, _ in patterns_data:
    risk_counts[risk] += 1

print("\nPattern distribution by risk:")
for risk, count in risk_counts.items():
    print(f"{risk}: {count} ({count/num_patterns*100:.1f}%)")

# Save patterns to CSV
with open('sql_injection_patterns.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['pattern', 'risk', 'score'])
    for pattern_data in patterns_data:
        writer.writerow(pattern_data)

# Display a sample of patterns (first 10)
print("\nSample of generated patterns:")
for i, (pattern, risk, score) in enumerate(patterns_data[:10]):
    print(f"{i+1}. \"{pattern}\" - {risk} ({score})")

# Generate a pattern file for C++ code
with open('patterns.txt', 'w') as f:
    f.write("{\n")
    for i, (pattern, _, _) in enumerate(patterns_data):
        if i < len(patterns_data) - 1:
            f.write(f'    "{pattern}",\n')
        else:
            f.write(f'    "{pattern}"\n')
    f.write("}\n")

print("\nGenerated patterns.txt for C++ code inclusion")
print(f"Total of {num_patterns} patterns generated")