 x= [
    [2, 3],
    [1, 2],
    [8, 7],
    [9, 8]
]

y = [0, 0, 1, 1]

def gini(y):
    classes =set(y)
    impurity =1
    for c in classes:
        p=y.count(c)/len(y)
        impurity-=p**2
    return impurity

print(gini(y))

def split_dataset(x,y, features,threshold):
    left_x=[]
    left_y=[]
    right_x=[]
    right_y=[]
    for i in range(len(x)):
        if x[i][features]<threshold:
            left_x.append(x[i])
            left_y.append(y[i])
        else:
            right_x.append(x[i])
            right_y.append(y[i])

    return left_x,left_y,right_x,right_y

left_x,left_y, right_x, right_y =split_dataset(x,y,0,5)
print(left_y,right_y)

def wt_gini(left_y,right_y):
    n=len(left_y)+len(right_y)
    wt=(
        (len(left_y)/n)*gini(left_y)
        +
        (len(right_y)/n)*gini(right_y)
        )
    return wt

print(wt_gini(left_y,right_y))

import random
def rndm_f(n_f,max_f):
    feature=list(range(n_f))
    return random.sample(feature,max_f)

print(rndm_f(4,2))

def best_split(x,y,max_f):
    b_g=float("inf")
    b_f=None
    b_t=None
    features= rndm_f(len(x[0]),max_f)

    for feature in features:
        for row in x:
            threshold = row[feature]
            left_x,left_y,right_x,right_y=split_dataset(x,y,feature,threshold)
            if(len(left_y)==0) or len(right_y)==0:
                continue
            c_g= wt_gini(left_y,right_y)
            if c_g<b_g:
                b_g=c_g
                b_f=feature
                b_t=threshold
    return b_f,b_t,b_g

feature, threshold, score = best_split(x,y,1)
print(feature,score,threshold)

def maj_class(y):
    c={}
    for v in y:
        if v not in c:
            c[v]=0
        c[v] +=1
    return max(c, key=c.get)

print(maj_class(y))

def build_tree(x,y,max_f):
    if (len(set(y)))==1:
        return {
            "leaf": y[0]
            }
    feature,threshold,score=best_split(x,y,max_f)
    left_x,left_y,right_x,right_y=split_dataset(x,y,feature,threshold)
    left_stree=build_tree(left_x,left_y,max_f)
    right_stree=build_tree(right_x,right_y,max_f)

    return{
        "feature":feature,
        "threshold":threshold,
        "left":left_stree,
        "right":right_stree
    }

print(build_tree(x,y,1))

def predict_tree(tree,sample):
    if "leaf" in tree:
        return tree["leaf"]
    
    feature =tree["feature"]
    threshold =tree["threshold"]

    if sample[feature]<threshold:
        return predict_tree(tree["left"], sample)
    else:
        return predict_tree(tree["right"],sample)

tree=build_tree(x,y,1)
print(predict_tree(tree,[6,5]))

def b_sample(x,y):
    sample_x=[]
    sample_y=[]

    for i in range(len(x)):
        index= random.randint(0,len(x)-1)
        sample_x.append(x[index])
        sample_y.append(y[index])

    return sample_x,sample_y

sx,sy=(b_sample(x,y))
print(sx,"\n",sy)

def forest(x,y,n_tree,max_f):
    forest=[]

    for i in range(n_tree):
        sx,sy=b_sample(x,y)
        tree=build_tree(sx,sy,max_f)
        forest.append(tree)

    return forest

forest = forest(x,y,5,1)
print(forest)

def p_forest(forest, sample):
    p=[]

    for tree in forest:
        pr= predict_tree(tree,sample)
        p.append(pr)

    return p

print(p_forest(forest,[6,5]))

def fp(forest,sample):
    pred=p_forest(forest,sample)
    return maj_class(pred)

print (fp(forest,[6,5]))