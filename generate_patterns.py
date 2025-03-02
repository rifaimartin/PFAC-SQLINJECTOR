import random
import string

# Base patterns untuk tiap kategori
union_patterns = [
    'UNION SELECT',
    'UNION ALL SELECT',
    'UNION/**/SELECT',
    'UNION+SELECT',
    'UNION%20SELECT'
]

error_patterns = [
    'OR 1=1',
    "OR '1'='1",
    'OR "1"="1"',
    'OR 1 IN (1)',
    'OR TRUE'
]

boolean_patterns = [
    'AND 1=1',
    'AND TRUE',
    'AND FALSE',
    "1' OR '1'='1",
    '1=1'
]

time_patterns = [
    'SLEEP(',
    'BENCHMARK(',
    'WAITFOR DELAY',
    'pg_sleep(',
    'DELAY'
]

stacked_patterns = [
    ';SELECT',
    ';INSERT',
    ';UPDATE',
    ';DELETE',
    ';DROP'
]

comment_patterns = [
    '--',
    '/*',
    '*/',
    '#',
    '//'
]

# Dictionary untuk mapping tipe pattern
pattern_types = {
    'UNION_BASED': union_patterns,
    'ERROR_BASED': error_patterns,
    'BOOLEAN_BASED': boolean_patterns,
    'TIME_BASED': time_patterns,
    'STACKED_QUERIES': stacked_patterns,
    'COMMENT_BASED': comment_patterns
}

def generate_variation(base_pattern):
    """Generate variasi dari pattern dasar"""
    variations = []
    # Case variations
    variations.append(base_pattern.lower())
    variations.append(base_pattern.upper())
    
    # Space variations
    variations.append(base_pattern.replace(' ', '+'))
    variations.append(base_pattern.replace(' ', '%20'))
    variations.append(base_pattern.replace(' ', '/**/'))
    
    # Add random characters
    chars = string.ascii_letters + string.digits
    variations.append(base_pattern + random.choice(chars))
    variations.append(random.choice(chars) + base_pattern)
    
    return variations

def generate_sql_patterns(num_patterns):
    """Generate SQL injection patterns"""
    patterns = []
    patterns_per_type = min(num_patterns // len(pattern_types), 100)  # Batasi maksimal 100 pattern per tipe
    
    for pattern_type, base_patterns in pattern_types.items():
        count = 0
        while count < patterns_per_type and count < 100:  # Tambah batas count
            base = random.choice(base_patterns)
            variations = generate_variation(base)
            for var in variations[:3]:  # Batasi hanya 3 variasi per pattern
                if count >= patterns_per_type:
                    break
                patterns.append("{}|{}|{}".format(pattern_type, var, random.randint(1,3)))
                count += 1
                if len(patterns) >= num_patterns:  # Batasi total patterns
                    return patterns
    
    return patterns

def generate_benign_queries(num_queries):
    """Generate benign SQL queries"""
    tables = ['users', 'products', 'orders', 'customers', 'categories']
    columns = ['id', 'name', 'email', 'price', 'quantity', 'status', 'created_at']
    conditions = ['>', '<', '=', '>=', '<=']
    
    queries = []
    for _ in range(num_queries):
        query_type = random.choice(['SELECT', 'INSERT', 'UPDATE', 'DELETE'])
        
        if query_type == 'SELECT':
            cols = random.sample(columns, random.randint(1, 3))
            table = random.choice(tables)
            query = f"SELECT {', '.join(cols)} FROM {table}"
            if random.random() > 0.5:
                col = random.choice(columns)
                cond = random.choice(conditions)
                value = random.randint(1, 1000)
                query += f" WHERE {col} {cond} {value}"
        
        elif query_type == 'INSERT':
            table = random.choice(tables)
            cols = random.sample(columns, random.randint(1, 3))
            values = [str(random.randint(1, 1000)) for _ in cols]
            query = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join(values)})"
        
        elif query_type == 'UPDATE':
            table = random.choice(tables)
            col = random.choice(columns)
            value = random.randint(1, 1000)
            query = f"UPDATE {table} SET {col} = {value} WHERE id = {random.randint(1, 100)}"
        
        else:  # DELETE
            table = random.choice(tables)
            query = f"DELETE FROM {table} WHERE id = {random.randint(1, 100)}"
        
        queries.append(query)
    
    return queries

def generate_malicious_queries(num_queries, patterns):
    """Generate malicious SQL queries using patterns"""
    benign_base = generate_benign_queries(num_queries)
    malicious = []
    
    for query in benign_base:
        # Pilih random pattern
        pattern_line = random.choice(patterns)
        _, pattern, _ = pattern_line.split('|')
        
        # Insert pattern ke query
        insertion_point = random.randint(0, len(query))
        malicious_query = query[:insertion_point] + ' ' + pattern + ' ' + query[insertion_point:]
        malicious.append(malicious_query)
    
    return malicious

def main():
    # Generate patterns dan queries
    num_patterns = 1000
    num_queries = 200

    print("Generating patterns and queries...")

    # Generate dan save patterns
    patterns = generate_sql_patterns(num_patterns)
    with open('data/sql_patterns.txt', 'w') as f:
        f.write('# SQL Injection Patterns\n')
        for pattern in patterns:
            f.write(pattern + '\n')

    # Generate dan save benign queries
    benign = generate_benign_queries(num_queries)
    with open('data/test_cases/benign_queries.txt', 'w') as f:
        f.write('# Benign SQL Queries\n')
        for query in benign:
            f.write(query + '\n')

    # Generate dan save malicious queries
    malicious = generate_malicious_queries(num_queries, patterns)
    with open('data/test_cases/malicious_queries.txt', 'w') as f:
        f.write('# Malicious SQL Queries\n')
        for query in malicious:
            f.write(query + '\n')

    print(f"Generated:")
    print(f"- {len(patterns)} patterns")
    print(f"- {len(benign)} benign queries")
    print(f"- {len(malicious)} malicious queries")

    # Print samples
    print("\nSample Patterns:")
    for pattern in patterns[:5]:
        print(pattern)

    print("\nSample Benign Queries:")
    for query in benign[:5]:
        print(query)

    print("\nSample Malicious Queries:")
    for query in malicious[:5]:
        print(query)

if __name__ == "__main__":
    main()