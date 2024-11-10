import csv
from itertools import combinations

# Load transactions from CSV file
filename = 'apriori.csv'
transactions = []
with open(filename, newline='') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        transactions.append(row)

# Set minimum support and confidence
min_support = 2
min_confidence = 0.7

# Step 1: Get C1 (candidate set of size 1)
C1 = []
for transaction in transactions:
    for item in transaction:
        if [item] not in C1:
            C1.append([item])

# Step 2: Count frequency of each item in C1 and prune non-frequent items
L1 = []
support_count = {}  # Track support count for each itemset
for candidate in C1:
    count = 0
    for transaction in transactions:
        found = True
        for item in candidate:
            if item not in transaction:
                found = False
                break
        if found:
            count += 1
    if count >= min_support:
        L1.append(candidate)
        support_count[tuple(candidate)] = count

# Store frequent itemsets by level
frequent_itemsets_by_level = []
frequent_itemsets_by_level.append(L1)

# Step 3 and 4: Generate larger itemsets (Lk) and continue until no larger itemsets are found
k = 2
Lk = L1
while Lk:
    # Generate candidates Ck from Lk (previous level's frequent itemsets)
    Ck = []
    for i in range(len(Lk)):
        for j in range(i + 1, len(Lk)):
            candidate = list(set(Lk[i]) | set(Lk[j]))
            candidate.sort()
            if len(candidate) == k and candidate not in Ck:
                Ck.append(candidate)

    # Count frequency for each candidate in Ck and prune those below min_support
    Lk = []
    for candidate in Ck:
        count = 0
        for transaction in transactions:
            found = True
            for item in candidate:
                if item not in transaction:
                    found = False
                    break
            if found:
                count += 1
        if count >= min_support:
            Lk.append(candidate)
            support_count[tuple(candidate)] = count

    # Append Lk to frequent itemsets by level and prepare for next level
    if Lk != []:
        frequent_itemsets_by_level.append(Lk)
        k += 1

# Generate association rules from frequent itemsets
association_rules = []
for level_itemsets in frequent_itemsets_by_level:
    for itemset in level_itemsets:
        if len(itemset) > 1:
            # Generate all non-empty subsets of itemset
            subsets = []
            for r in range(1, len(itemset)):
                for subset in combinations(itemset, r):
                    subsets.append(subset)
            
            itemset_support = support_count[tuple(itemset)]
            
            # For each subset, calculate confidence and create rule if confidence >= min_confidence
            for subset in subsets:
                subset_support = support_count.get(subset, 0)
                if subset_support > 0:
                    confidence = itemset_support / subset_support
                    if confidence >= min_confidence:
                        remaining = list(set(itemset) - set(subset))
                        remaining.sort()
                        rule = (subset, remaining, confidence)
                        association_rules.append(rule)

# Output the frequent itemsets by level
print("Frequent Itemsets by Level:")
level_count = 1  # Manually tracking the level count
for itemsets in frequent_itemsets_by_level:
    print(f"Frequent Itemset {level_count}: {itemsets}")
    level_count += 1

# Output the association rules
print("\nAssociation Rules:")
for rule in association_rules:
    antecedent, consequent, confidence = rule
    print(f"{antecedent} => {consequent} (confidence: {confidence:.2f})")
