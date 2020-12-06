a=[32,3, 9,4, 7, 8,  31, 57, 66,0, 1]
# a=  [5, 6, 7, 9, 1, 2,4]

def find_mini_heap(tree):
    """
    该方法没有用分治
    :param tree:
    :return:
    """
    len_tree=len(tree)
    for i in range(int(len(tree)/2)-1,-1,-1):
        if (2*(i+1)-1<=len_tree-1) and (tree[2*(i+1)-1]<tree[i]):
            tree[2 * (i + 1) - 1],tree[i]=tree[i],tree[2*(i+1)-1]
        if (2*(i+1)<=len_tree-1) and (tree[2*(i+1)]<tree[i]):
            tree[2 * (i + 1)],tree[i]=tree[i],tree[2*(i+1)]
    return a[0]
def find_mini(tree,root):
    if 2 * (root + 1) - 1 >= len(tree):  # the leaf
        return root
    elif 2 * (root + 1) >= len(tree):  # only left child
        if tree[root] > tree[2 * (root + 1) - 1]:
            return 2 * (root + 1) - 1
        else:
            return root
    else:  # the root has two children
        if tree[root] < tree[2 * (root + 1) - 1] and tree[root] < tree[2 * (root + 1)]:
            return root
        elif not (tree[root] < tree[2 * (root + 1) - 1]) and tree[root] < tree[2 * (root + 1)]:
            return find_mini(tree, 2 * (root + 1) - 1)
        else:
            # tree[root] <tree[2 * (root + 1) - 1] and not (tree[root]<tree[2*(root+1)]):
            return find_mini(tree, 2 * (root + 1))
local_pos=find_mini(a,0)
print(a[local_pos])