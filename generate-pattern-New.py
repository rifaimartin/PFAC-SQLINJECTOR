import random
import string
import csv
import os

def generate_random_string(length):
    """Generate a random string of lowercase letters and digits"""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

def create_patterns_file():
    """Create a patterns.txt file with optimized SQL injection patterns"""
    patterns = [
        # CRITICAL RISK PATTERNS (Weight 90)
        "xp_cmdshell",
        "exec xp_",
        "into outfile",
        "load_file",
        "load data infile",
        "/etc/passwd",
        "shell.php",
        "net user hack",
        "lambda_async",
        "create user",
        "grant all privileges",
        "create trigger",
        
        # HIGH RISK PATTERNS (Weight 50)
        "drop table",
        "drop database",
        "drop procedure",
        "delete from",
        "alter table",
        "insert into",
        "update users",
        "exec(@cmd)",
        "exec sp_",
        "exec master",
        "pg_sleep",
        "sleep(",
        "waitfor delay",
        "benchmark(",
        "@@version",
        "mysql.user",
        
        # MEDIUM-HIGH RISK PATTERNS (Weight 30)
        "union select",
        "union all select",
        "information_schema",
        "password from users",
        "from users",
        "table_name",
        "table_schema",
        "count(*)",
        "group by",
        "having",
        
        # MEDIUM RISK PATTERNS (Weight 18)
        "union",
        "select from",
        "' select",
        "username",
        "admin' --",
        "convert(",
        "cast(",
        "md5(",
        "limit 1",
        
        # LOW-MEDIUM RISK PATTERNS (Weight 10)
        "1=1",
        "' or",
        "\" or",
        "' and",
        "\" and",
        "'='",
        "\"=\"",
        "true=true",
        "' ||",
        "\" ||",
        "'a'='a",
        "'x'='x",
        "'1'='1",
        
        # LOW RISK PATTERNS (Weight 5)
        "--",
        "/**/",
        "/* test */",
        "' =",
        "\" =",
        " <=",
        " >=",
        " <>",
        "= =",
        "= <",
        "= or",
        "= ||"
    ]

    # Write patterns to file in proper format
    with open('patterns.txt', 'w') as f:
        for pattern in patterns:
            f.write(f'"{pattern}",\n')

    print(f"Created patterns.txt with {len(patterns)} patterns")

def get_query_templates():
    """Define SQL injection query templates by risk level aligned with pattern weights"""
    templates = {
        'low': [
            # Queries that should score 5-20 points (LOW classification)
            "' OR 1=1 --",                    # Contains: "' or", "1=1", "--" = 10+10+5 = 25 -> but threshold is 20, so some will be medium
            "\" OR \"1\"=\"1",               # Contains: "\" or", "\"=\"" = 10+10 = 20
            "' OR 'a'='a",                   # Contains: "' or", "'a'='a" = 10+10 = 20  
            "admin' --",                     # Contains: "admin' --", "--" = 18+5 = 23 -> medium
            "' || 1=1 --",                   # Contains: "' ||", "1=1", "--" = 10+10+5 = 25 -> medium
            "1' OR '1'='1",                  # Contains: "' or", "'1'='1" = 10+10 = 20
            "' OR 'x'='x",                   # Contains: "' or", "'x'='x" = 10+10 = 20
            "OR 1=1",                        # Contains: "1=1" = 10 only
            "' = 1",                         # Contains: "' =" = 5 only
            "\" = 1",                        # Contains: "\" =" = 5 only
            "' <> 1",                        # Contains: " <>" = 5 only
            "' >= 1",                        # Contains: " >=" = 5 only
            "' <= 1",                        # Contains: " <=" = 5 only
        ],
        'medium': [
            # Queries that should score 21-45 points (MEDIUM classification)
            "' UNION SELECT 1,2 --",                    # Contains: "union", "' select", "--" = 18+18+5 = 41
            "' OR 1=1 UNION SELECT 1 --",              # Contains: "' or", "1=1", "union", "' select", "--" = 10+10+18+18+5 = 61 -> HIGH
            "'; SELECT @@version --",                   # Contains: "' select", "@@version", "--" = 18+50+5 = 73 -> HIGH
            "' HAVING 1=1 #",                          # Contains: "having", "1=1" = 30+10 = 40
            "' GROUP BY 1 --",                         # Contains: "group by", "--" = 30+5 = 35
            "' AND username LIKE 'admin' --",          # Contains: "' and", "username", "--" = 10+18+5 = 33
            "1' OR admin' --",                         # Contains: "' or", "admin' --", "--" = 10+18+5 = 33
            "' UNION ALL SELECT 1 --",                 # Contains: "union", "' select", "--" = 18+18+5 = 41
            "' OR MD5('test') --",                     # Contains: "' or", "md5(", "--" = 10+18+5 = 33
            "' OR CONVERT(int, '1') --",               # Contains: "' or", "convert(", "--" = 10+18+5 = 33
            "' OR CAST('1' as int) --",                # Contains: "' or", "cast(", "--" = 10+18+5 = 33
            "' LIMIT 1 /**/",                          # Contains: "limit 1", "/**/" = 18+5 = 23
        ],
        'high': [
            # Queries that should score 46-75 points (HIGH classification)
            "'; DROP TABLE temp --",                        # Contains: "drop table", "--" = 50+5 = 55
            "'; DELETE FROM users --",                      # Contains: "delete from", "--" = 50+5 = 55
            "'; ALTER TABLE users ADD col --",              # Contains: "alter table", "--" = 50+5 = 55
            "'; INSERT INTO logs VALUES('hack') --",        # Contains: "insert into", "--" = 50+5 = 55
            "'; UPDATE users SET pass='hack' --",           # Contains: "update users", "--" = 50+5 = 55
            "'; DROP PROCEDURE admin_login --",             # Contains: "drop procedure", "--" = 50+5 = 55
            "'; DROP DATABASE temp --",                     # Contains: "drop database", "--" = 50+5 = 55
            "'; EXEC sp_configure 'options', 1 --",         # Contains: "exec sp_", "--" = 50+5 = 55
            "'; WAITFOR DELAY '0:0:5' --",                  # Contains: "waitfor delay", "--" = 50+5 = 55
            "'; SELECT PG_SLEEP(5) --",                     # Contains: "pg_sleep", "' select", "--" = 50+18+5 = 73
            "'; SELECT SLEEP(5) --",                        # Contains: "sleep(", "' select", "--" = 50+18+5 = 73
            "'; SELECT BENCHMARK(10000,MD5('x')) --",       # Contains: "benchmark(", "' select", "md5(", "--" = 50+18+18+5 = 91 -> CRITICAL
            "' UNION SELECT @@version --",                  # Contains: "union select", "@@version", "--" = 30+50+5 = 85 -> CRITICAL
            "' OR 1=1; DROP TABLE users --",                # Contains: "' or", "1=1", "drop table", "--" = 10+10+50+5 = 75
            "'; EXEC master..xp_cmdshell 'ping' --",        # Contains: "exec master", "--" = 50+5 = 55
        ],
        'critical': [
            # Queries that should score 76+ points (CRITICAL classification)
            "'; EXEC xp_cmdshell('net user hack') --",            # Contains: "exec xp_", "net user hack", "--" = 90+90+5 = 185
            "'; SELECT load_file('/etc/passwd') --",              # Contains: "load_file", "/etc/passwd", "' select", "--" = 90+90+18+5 = 203
            "' INTO OUTFILE '/var/www/shell.php' --",             # Contains: "into outfile", "shell.php", "--" = 90+90+5 = 185
            "'; LOAD DATA INFILE '/etc/passwd' INTO TABLE users --", # Contains: "load data infile", "/etc/passwd", "insert into", "--" = 90+90+50+5 = 235
            "'; CREATE USER 'hack'@'%' IDENTIFIED BY 'pass' --", # Contains: "create user", "--" = 90+5 = 95
            "'; GRANT ALL PRIVILEGES ON *.* TO 'hack' --",       # Contains: "grant all privileges", "--" = 90+5 = 95
            "'; CREATE TRIGGER evil BEFORE INSERT ON users --",  # Contains: "create trigger", "insert into", "--" = 90+50+5 = 145
            "'; CALL mysql.lambda_async('127.0.0.1', 1234) --",  # Contains: "lambda_async", "--" = 90+5 = 95
            "'; SELECT * FROM mysql.user --",                    # Contains: "mysql.user", "' select", "from users", "--" = 50+18+30+5 = 103
            "'; UNION SELECT username, password FROM users --",   # Contains: "union select", "password from users", "--" = 30+30+5 = 65 -> HIGH (need to adjust)
            "'; SELECT table_name FROM information_schema.tables --", # Contains: "' select", "table_name", "information_schema", "--" = 18+30+30+5 = 83
            "'; SELECT COUNT(*) FROM information_schema.tables --",   # Contains: "' select", "count(*)", "information_schema", "--" = 18+30+30+5 = 83
            "'; EXEC(@cmd) --",                                   # Contains: "exec(@cmd)", "--" = 50+5 = 55 -> HIGH (need critical pattern)
            "' UNION ALL SELECT username, password FROM users --", # Contains: "union all select", "password from users", "--" = 30+30+5 = 65 -> HIGH
        ]
    }
    return templates

def calculate_expected_score(query, patterns):
    """Calculate expected score based on patterns found in query"""
    query_lower = query.lower()
    total_score = 0
    matched_patterns = []
    
    # Pattern weights based on our weight assignment
    pattern_weights = {
        # Critical (90)
        "xp_cmdshell": 90, "exec xp_": 90, "into outfile": 90, "load_file": 90,
        "load data infile": 90, "/etc/passwd": 90, "shell.php": 90, "net user hack": 90,
        "lambda_async": 90, "create user": 90, "grant all privileges": 90, "create trigger": 90,
        
        # High (50)
        "drop table": 50, "drop database": 50, "drop procedure": 50, "delete from": 50,
        "alter table": 50, "insert into": 50, "update users": 50, "exec(@cmd)": 50,
        "exec sp_": 50, "exec master": 50, "pg_sleep": 50, "sleep(": 50,
        "waitfor delay": 50, "benchmark(": 50, "@@version": 50, "mysql.user": 50,
        
        # Medium-High (30)
        "union select": 30, "union all select": 30, "information_schema": 30,
        "password from users": 30, "from users": 30, "table_name": 30,
        "table_schema": 30, "count(*)": 30, "group by": 30, "having": 30,
        
        # Medium (18)
        "union": 18, "select from": 18, "' select": 18, "username": 18,
        "admin' --": 18, "convert(": 18, "cast(": 18, "md5(": 18, "limit 1": 18,
        
        # Low-Medium (10)
        "1=1": 10, "' or": 10, "\" or": 10, "' and": 10, "\" and": 10,
        "'='": 10, "\"=\"": 10, "true=true": 10, "' ||": 10, "\" ||": 10,
        "'a'='a": 10, "'x'='x": 10, "'1'='1": 10,
        
        # Low (5)
        "--": 5, "/**/": 5, "/* test */": 5, "' =": 5, "\" =": 5,
        " <=": 5, " >=": 5, " <>": 5, "= =": 5, "= <": 5, "= or": 5, "= ||": 5
    }
    
    for pattern, weight in pattern_weights.items():
        if pattern in query_lower:
            total_score += weight
            matched_patterns.append(f"{pattern}({weight})")
    
    return total_score, matched_patterns

def adjust_query_for_target_score(base_query, target_min, target_max, patterns):
    """Adjust query to hit target score range"""
    current_score, matched = calculate_expected_score(base_query, patterns)
    
    if target_min <= current_score <= target_max:
        return base_query, current_score
    
    # If score is too low, add more patterns
    if current_score < target_min:
        # Add patterns to increase score
        additional_patterns = {
            'low': [" AND 1=1", " OR 'a'='a", " /**/"],
            'medium': [" UNION SELECT 1", " GROUP BY 1", " HAVING 1=1"],
            'high': [" AND @@version IS NOT NULL", " OR SLEEP(1)", " OR BENCHMARK(1,MD5('x'))"],
            'critical': [" AND load_file('test')", " OR xp_cmdshell"]
        }
        
        # Determine which level we need
        if target_max <= 20:
            level = 'low'
        elif target_max <= 45:
            level = 'medium'
        elif target_max <= 75:
            level = 'high'
        else:
            level = 'critical'
        
        # Add a random pattern from appropriate level
        if level in additional_patterns:
            addition = random.choice(additional_patterns[level])
            modified_query = base_query + addition
            new_score, _ = calculate_expected_score(modified_query, patterns)
            if target_min <= new_score <= target_max:
                return modified_query, new_score
    
    # If score is too high, try to use a simpler version
    if current_score > target_max:
        # Try to simplify the query
        simple_queries = {
            'low': ["' OR 1=1", "admin' --", "' = 1"],
            'medium': ["' UNION SELECT 1", "' GROUP BY 1", "' HAVING 1=1"],
            'high': ["'; DROP TABLE temp", "'; DELETE FROM users", "' OR @@version"],
            'critical': ["'; EXEC xp_cmdshell('dir')", "' INTO OUTFILE '/tmp/test'"]
        }
        
        if target_max <= 20:
            level = 'low'
        elif target_max <= 45:
            level = 'medium'
        elif target_max <= 75:
            level = 'high'
        else:
            level = 'critical'
        
        if level in simple_queries:
            simple_query = random.choice(simple_queries[level])
            new_score, _ = calculate_expected_score(simple_query, patterns)
            if target_min <= new_score <= target_max:
                return simple_query, new_score
    
    # Return original if we can't adjust properly
    return base_query, current_score

def generate_dataset(risk_level, size, num_queries, patterns):
    """Generate dataset for a specific risk level with accurate scoring"""
    templates = get_query_templates()[risk_level]
    
    # Define target score ranges for each risk level
    target_ranges = {
        'low': (1, 20),
        'medium': (21, 45),
        'high': (46, 75),
        'critical': (76, 200)
    }
    
    target_min, target_max = target_ranges[risk_level]
    queries = []

    print(f"\nGenerating {risk_level} dataset...")
    score_distribution = []

    for i in range(num_queries):
        # Choose a random template
        template = random.choice(templates)
        
        # Adjust query to hit target score range
        final_query, actual_score = adjust_query_for_target_score(
            template, target_min, target_max, patterns
        )
        
        score_distribution.append(actual_score)
        queries.append((final_query, risk_level, actual_score))
        
        # Print first few examples
        if i < 3:
            expected_score, matched_patterns = calculate_expected_score(final_query, patterns)
            print(f"  Query {i+1}: {final_query}")
            print(f"    Expected score: {expected_score}")
            print(f"    Matched patterns: {', '.join(matched_patterns)}")

    # Print score statistics
    if score_distribution:
        avg_score = sum(score_distribution) / len(score_distribution)
        min_score = min(score_distribution)
        max_score = max(score_distribution)
        print(f"  Score range: {min_score}-{max_score}, Average: {avg_score:.1f}")

    return queries

def generate_all_datasets():
    """Generate all datasets with accurate pattern-based scoring"""
    os.makedirs('datasets', exist_ok=True)

    # Load patterns for score calculation
    patterns = []
    try:
        with open('patterns.txt', 'r') as f:
            for line in f:
                line = line.strip()
                if line and line.startswith('"') and line.endswith('",'):
                    pattern = line[1:-2]  # Remove quotes and comma
                    patterns.append(pattern)
    except FileNotFoundError:
        print("patterns.txt not found, creating it first...")
        create_patterns_file()
        return generate_all_datasets()

    # Dataset sizes to generate
    sizes = [10, 100, 1000, 5000, 10000]

    # Generate for each risk level
    for risk_level in ['low', 'medium', 'high', 'critical']:
        for size in sizes:
            # Number of queries to generate
            num_queries = min(size, 10000)

            # Generate dataset
            queries = generate_dataset(risk_level, size, num_queries, patterns)

            # Write to CSV
            filename = f'datasets/sql_dataset_{risk_level.capitalize()}_{size}.csv'
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['query', 'risk'])
                for query, risk, score in queries:
                    writer.writerow([query, risk])

            print(f"Created {filename} with {num_queries} queries")

if __name__ == "__main__":
    # Create patterns file
    create_patterns_file()

    # Generate all datasets
    generate_all_datasets()