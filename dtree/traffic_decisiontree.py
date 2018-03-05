my_data=[[1,   8,  'sunny ',  'no', 'no' ,'long']
        [2,   8,   'cloudy' , 'no', 'yes' ,'long']
        [3,   10,  'sunny ',  'no', 'no' ,'short']
        [4,   9,   'rainy ',  'yes', 'no' ,'long']
        [5,   9,   'sunny ',  'yes', 'yes' ,'long']
        [6,   10,  'sunny ',  'no', 'no' ,'short']
        [7,   10,  'cloudy' , 'no', 'no' ,'short']
        [8,   9,   'rainy ',  'no', 'no' ,'medium']
        [9,   9,   'sunny ',  'yes', 'no' ,'long']
        [10,  10,  'cloudy' , 'yes', 'yes' ,'long']
        [11,  10,  'rainy ',  'no', 'no' ,'short']
        [12,  8,   'cloudy' , 'yes', 'no' ,'long']
        [13,  9,   'sunny ',  'no', 'no' ,'medium']]


class decisionnode:
    def __init__(self,col=-1,value=None,results=None,tb=None,fb=None):
        self.col=col # column index of criteria being tested
        self.value=value # vlaue necessary to get a true result
        self.results=results # dict of results for a branch, None for everything except endpoints
        self.tb=tb # true decision nodes
        self.fb=fb # false decision nodes


# Divides a set on a specific column. Can handle numeric or nominal values
def divideset(rows, column, value):
    # for numerical values
    if isinstance(value, int) or isinstance(value, float):
        split_function = lambda row: row[column] >= value
    # for nominal values
    else:
        split_function = lambda row: row[column] == value
    # Divide the rows into two sets and return them
    set1 = [row for row in rows if split_function(row)]  # if split_function(row)
    set2 = [row for row in rows if not split_function(row)]
    return (set1, set2)


# Create counts of possible results (last column of each row is the result)
def uniquecounts(rows):
    results={}
    for row in rows:
        # The result is the last column
        r=row[len(row)-1]
        if r not in results: results[r]=0
        results[r]+=1
    return results

from collections import defaultdict
def uniquecounts_dd(rows):
    results = defaultdict(lambda: 0)
    for row in rows:
        r = row[len(row)-1]
        results[r]+=1
    return dict(results)


# Entropy is the sum of p(x)log(p(x)) across all the different possible results
# p(i) = frequency(outcome) = count(outcome) / count(total rows)
# entropy = sum of p(i) * log(p(i))

def entropy(rows):
    from math import log
    log2=lambda x:log(x)/log(2)
    results=uniquecounts(rows)
    # calculate the entropy
    ent=0.0
    for r in results.keys():
        # current probability of class
        p=float(results[r])/len(rows)
        ent=ent-p*log2(p)
    return ent

my_data2=[[1,   8,   'cloudy',  'no',  'no']
        [2,   8,   'sunny',   'yes', 'no']
        [3,   8,   'rainy',   'no',  'yes']
        [4,   9,   'rainy',   'yes', 'no']
        [5,   9,   'cloudy',  'yes', 'yes']
        [6,   9,   'sunny',   'no',  'no']
        [7,   10,  'sunny',   'yes', 'yes']
        [8,   10,  'cloudy',  'no',  'yes']
        [9,   10,  'rainy',   'no',  'no']]

print 'entropy:', entropy(my_data2)


def buildtree(rows, scorefun=entropy):
    if len(rows) == 0: return decisionnode()
    current_score = scorefun(rows)

    best_gain = 0.0
    best_criteria = None
    best_sets = None

    column_count = len(rows[0]) - 1	 # last column is result
    for col in range(0, column_count):
        # find different values in this column
        column_values = set([row[col] for row in rows])
        # for each possible value, try to divide on that value
        for value in column_values:
            set1, set2 = divideset(rows, col, value)
            # Information gain
            p = float(len(set1)) / len(rows)
            gain = current_score - p*scorefun(set1) - (1-p)*scorefun(set2)
            if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set1, set2)

    if best_gain > 0:
        trueBranch = buildtree(best_sets[0])
        falseBranch = buildtree(best_sets[1])
        return decisionnode(col=best_criteria[0], value=best_criteria[1],
                tb=trueBranch, fb=falseBranch)
    else:
        return decisionnode(results=uniquecounts(rows))


def printtree(tree, indent=''):
    # Is this a leaf node?
    if tree.results!=None:
        print str(tree.results)
    else:
        # Print the criteria
        print 'Column ' + str(tree.col)+' : '+str(tree.value)+'? '

        print indent+'True->',
        printtree(tree.tb,indent+'  ')
        print indent+'False->',
        printtree(tree.fb,indent+'  ')

print(buildtree(my_data))