// Copyright 2016 Guiming Zhang (gzhang45@wisc.edu)
// Distributed under GNU General Public License (GPL) license

// Revised based on http://nghiaho.com/uploads/code/CUDA_KDtree.zip

#ifndef __KDTREE_H__
#define __KDTREE_H__

#include <vector>
#include <cstddef>

using namespace std;

#define KDTREE_DIM 2 // data dimensions

struct Point
{
    float coords[KDTREE_DIM];
};

class KDNode
{
public:
    KDNode()
    {
        parent = NULL;
        left = NULL;
        right = NULL;
        split_value = -1;
        _parent = -1;
        _left = -1;
        _right = -1;
    }

    int id; // for GPU
    int level;
    KDNode *parent, *left, *right;
    int _parent, _left, _right; // for GPU
    float split_value;
    vector <int> indexes; // index to points
};

class KDtree
{
public:
    KDtree();
    ~KDtree();
    void Create(vector <Point> &pts, int max_levels = 99 /* You can limit the search depth if you want */);
    //void Search(const Point &query, int *ret_index, float *ret_sq_dist);
    void SearchRange(const Point &query, float range, vector<int> &ret_index, vector<float> &ret_sq_dist);
    //void SearchRangeBruteForce(const Point &query, float range, vector<int> &ret_index, vector<float> &ret_sq_dist);
    int GetNumNodes() const { return m_id; }
    KDNode* GetRoot() const { return m_root; }

    static bool SortPoints(const int a, const int b);

public:
    vector <Point> *m_pts;
    KDNode *m_root;
    int m_current_axis;
    int m_levels;
    int m_cmps; // count how many comparisons were made in the tree for a query
    int m_id; // current node ID

    void Split(KDNode *cur, KDNode *left, KDNode *right);
    //void SearchAtNode(KDNode *cur, const Point &query, int *ret_index, float *ret_dist, KDNode **ret_node);
    //void SearchAtNodeRange(KDNode *cur, const Point &query, float range, int *ret_index, float *ret_dist);
    //void SearchAtNodeRange(KDNode *cur, const Point &query, float range, vector<int> &ret_index, vector<float> &ret_sq_dist);
    inline float Distance(const Point &a, const Point &b) const;
};

float KDtree::Distance(const Point &a, const Point &b) const
{
    float dx = a.coords[0] - b.coords[0];
    float dy = a.coords[1] - b.coords[1];
    return dx*dx + dy*dy;
}

#endif
