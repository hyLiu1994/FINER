import pandas as pd

data_sparse_kt = {
    'DKT': {
        'AS2015': 0.7271,
        'NIPS34': 0.7689,
        'POJ': 0.6089
    },
    'DKT+': {
        'AS2015': 0.7285,
        'NIPS34': 0.7696,
        'POJ': 0.6173
    },
    'KQN': {
        'AS2015': 0.7254,
        'NIPS34': 0.7684,
        'POJ': 0.6080
    },
    'DKVMN': {
        'AS2015': 0.7227,
        'NIPS34': 0.7673,
        'POJ': 0.6056
    },
    'ATKT': {
        'AS2015': 0.7245,
        'NIPS34': 0.7665,
        'POJ': 0.6075
    },
    'GKT': {
        'AS2015': 0.7258,
        'NIPS34': 0.7689,
        'POJ': 0.6070
    },
    'SAKT': {
        'AS2015': 0.7114,
        'NIPS34': 0.7517,
        'POJ': 0.6095
    },
    'SAINT': {
        'AS2015': 0.7026,
        'NIPS34': 0.7873,
        'POJ': 0.5563
    },
    'AKT': {
        'AS2015': 0.7281,
        'NIPS34': 0.8033,
        'POJ': 0.6281
    },
    'HAWKES': {
        'NIPS34': 0.7767
    },
    'IEKT': {
        'NIPS34': 0.8045
    },
    'sparseKT-soft': {
        'AS2015': 0.7379,
        'NIPS34': 0.8033,
        'POJ': 0.6323
    },
    'sparseKT-topK': {
        'AS2015': 0.7501,
        'NIPS34': 0.8043,
        'POJ': 0.6401
    }
}
data_simplekt = {
    'DKT': {
        'AS2009': 0.7541,
        'AL2005': 0.8149,
        'BD2006': 0.8015,
        'NIPS34': 0.7689,
        'Statics2011': 0.8222,
        'AS2015': 0.7271,
        'POJ': 0.6089
    },
    'DKT+': {
        'AS2009': 0.7547,
        'AL2005': 0.8156,
        'BD2006': 0.8020,
        'NIPS34': 0.7696,
        'Statics2011': 0.8279,
        'AS2015': 0.7285,
        'POJ': 0.6173
    },
    'DKT-F': {
        'AL2005': 0.8147,
        'BD2006': 0.7985,
        'NIPS34': 0.7733,
        'Statics2011': 0.7839,
        'POJ': 0.6030
    },
    'KQN': {
        'AS2009': 0.7477,
        'AL2005': 0.8027,
        'BD2006': 0.7936,
        'NIPS34': 0.7684,
        'Statics2011': 0.8232,
        'AS2015': 0.7254,
        'POJ': 0.6080
    },
    'LPKT': {
        'AS2009': 0.7814,
        'AL2005': 0.8274,
        'BD2006': 0.8055,
        'NIPS34': 0.8035
    },
    'IEKT': {
        'AS2009': 0.7861,
        'AL2005': 0.8416,
        'BD2006': 0.8125,
        'NIPS34': 0.8045
    },
    'DKVMN': {
        'AS2009': 0.7473,
        'AL2005': 0.8054,
        'BD2006': 0.7983,
        'NIPS34': 0.7673,
        'Statics2011': 0.8093,
        'AS2015': 0.7227,
        'POJ': 0.6056
    },
    'ATKT': {
        'AS2009': 0.7470,
        'AL2005': 0.7995,
        'BD2006': 0.7889,
        'NIPS34': 0.7665,
        'Statics2011': 0.8055,
        'AS2015': 0.7245,
        'POJ': 0.6075
    },
    'GKT': {
        'AS2009': 0.7424,
        'AL2005': 0.8110,
        'BD2006': 0.8046,
        'NIPS34': 0.7689,
        'Statics2011': 0.8040,
        'AS2015': 0.7258,
        'POJ': 0.6070
    },
    'SAKT': {
        'AS2009': 0.7246,
        'AL2005': 0.7880,
        'BD2006': 0.7740,
        'NIPS34': 0.7517,
        'Statics2011': 0.7965,
        'AS2015': 0.7114,
        'POJ': 0.6095
    },
    'SAINT': {
        'AS2009': 0.6958,
        'AL2005': 0.7775,
        'BD2006': 0.7781,
        'NIPS34': 0.7873,
        'Statics2011': 0.7599,
        'AS2015': 0.7026,
        'POJ': 0.5563
    },
    'AKT': {
        'AS2009': 0.7853,
        'AL2005': 0.8306,
        'BD2006': 0.8208,
        'NIPS34': 0.8033,
        'Statics2011': 0.8309,
        'AS2015': 0.7281,
        'POJ': 0.6281
    },
    'simpleKT': {
        'AS2009': 0.7744,
        'AL2005': 0.8254,
        'BD2006': 0.8160,
        'NIPS34': 0.8033,
        'Statics2011': 0.8199,
        'AS2015': 0.7248,
        'POJ': 0.6252
    }
}
# Calculate relative improvement over DKT baseline
def calc_relative_improvement(baseline, dkt):
    return ((1-dkt)-(1-baseline))/(1-dkt)

for data in [data_sparse_kt, data_simplekt]:
    # Store DKT baseline results 
    dkt_results = data['DKT']

    # Calculate relative improvements for each model and dataset
    relative_improvements = {}

    for model in data.keys():
        if model == 'DKT':
            continue
            
        relative_improvements[model] = {}
        for dataset in data[model].keys():
            if dataset in dkt_results:
                baseline = data[model][dataset]
                dkt = dkt_results[dataset]
                rel_imp = calc_relative_improvement(baseline, dkt)
                relative_improvements[model][dataset] = round(rel_imp * 100, 2) # Convert to percentage

    # Convert to pandas DataFrame for better display
    df = pd.DataFrame(relative_improvements).T
    df = df.fillna('-')
    
    print("\nRelative improvements over DKT (%)")
    print(df.to_string(float_format=lambda x: '%.2f' % x if isinstance(x, float) else x))
    print("\n")
