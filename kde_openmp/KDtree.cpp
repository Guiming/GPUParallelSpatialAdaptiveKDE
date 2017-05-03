// Copyright 2016 Guiming Zhang (gzhang45@wisc.edu)
// Distributed under GNU General Public License (GPL) license

// Revised based on http://nghiaho.com/uploads/code/CUDA_KDtree.zip

#include "KDtree.h"
#include <cstdio>
#include <algorithm>
#include <numeric>
#include <float.h>
#include <cmath>

// Eww global. Need this for the sort function
// A pointer to the class itself, allows the sort function to determine the splitting axis to sort by
static KDtree *myself = NULL;

KDtree::KDtree()
{
    myself = this;
    m_id = 0;
}

KDtree::~KDtree()
{
    // Delete all the ndoes
    vector <KDNode*> to_visit;

    to_visit.push_back(m_root);

    while(to_visit.size()) {
        vector <KDNode*> next_search;

        while(to_visit.size()) {
            KDNode *cur = to_visit.back();
            to_visit.pop_back();

            if(cur->left)
                next_search.push_back(cur->left);

            if(cur->right)
                next_search.push_back(cur->right);

            delete cur;
        }

        to_visit = next_search;
    }

    m_root = NULL;
}

void KDtree::Create(vector <Point> &pts, int max_levels)
{
    // rule of thumb determination of tree depth
    //max_levels = (int)(log(pts.size())/log(2) / 2) + 1;
    printf("max_levels = %d\n", max_levels);

    m_pts = &pts;
    m_levels = max_levels;

    m_root = new KDNode();
    m_root->id = m_id++;
    m_root->level = 0;
    m_root->indexes.resize(pts.size());

    for(unsigned int i=0; i < pts.size(); i++) {
        m_root->indexes[i] = i;
    }

    vector <KDNode*> to_visit;
    to_visit.push_back(m_root);

    while(to_visit.size()) {
        vector <KDNode*> next_search;

        while(to_visit.size()) {
            KDNode *node = to_visit.back();
            to_visit.pop_back();

            if(node->level < max_levels) {
                if(node->indexes.size() > 1) {
                    KDNode *left = new KDNode();
                    KDNode *right = new KDNode();

                    left->id = m_id++;
                    right->id = m_id++;

                    Split(node, left, right);

                    // Clear current indexes
                    {
                        vector <int> dummy;
                        node->indexes.swap(dummy);
                    }

                    node->left = left;
                    node->right = right;

                    node->_left = left->id;
                    node->_right = right->id;

                    if(left->indexes.size())
                        next_search.push_back(left);

                    if(right->indexes.size())
                        next_search.push_back(right);
                }
            }
        }

        to_visit = next_search;
    }
}

bool KDtree::SortPoints(const int a, const int b)
{
    vector <Point> &pts = *myself->m_pts;

    return pts[a].coords[myself->m_current_axis] < pts[b].coords[myself->m_current_axis];
}

void KDtree::Split(KDNode *cur, KDNode *left, KDNode *right)
{
    // Assume left/right nodes are created already

    vector <Point> &pts = *m_pts;
    m_current_axis = cur->level % KDTREE_DIM;;

    sort(cur->indexes.begin(), cur->indexes.end(), KDtree::SortPoints);

    int mid = cur->indexes[cur->indexes.size() / 2];
    cur->split_value = pts[mid].coords[m_current_axis];

    left->parent = cur;
    right->parent = cur;

    left->level = cur->level+1;
    right->level = cur->level+1;

    left->_parent = cur->id;
    right->_parent = cur->id;

    for(unsigned int i=0; i < cur->indexes.size(); i++) {
        int idx = cur->indexes[i];

        if(pts[idx].coords[m_current_axis] < cur->split_value)
            left->indexes.push_back(idx);
        else
            right->indexes.push_back(idx);
    }
}
/*
void KDtree::SearchAtNode(KDNode *cur, const Point &query, int *ret_index, float *ret_dist, KDNode **ret_node)
{
    int best_idx = 0;
    float best_dist = FLT_MAX;
    vector <Point> &pts = *m_pts;

   // First pass
    while(true) {
        int split_axis = cur->level % KDTREE_DIM;

        m_cmps++;

        if(cur->left == NULL) {
            *ret_node = cur;

            for(unsigned int i=0; i < cur->indexes.size(); i++) {
                m_cmps++;

                int idx = cur->indexes[i];

                float dist = Distance(query, pts[idx]);

                if(dist < best_dist) {
                    best_dist = dist;
                    best_idx = idx;
                }
            }

            break;
        }
        else if(query.coords[split_axis] < cur->split_value) {
            cur = cur->left;
        }
        else {
            cur = cur->right;
        }
    }

    *ret_index = best_idx;
    *ret_dist = best_dist;
}


void KDtree::SearchAtNodeRange(KDNode *cur, const Point &query, float range, int *ret_index, float *ret_dist)
{
    int best_idx = 0;
    float best_dist = FLT_MAX;
    vector <Point> &pts = *m_pts;
    vector <KDNode*> to_visit;

    to_visit.push_back(cur);

    while(to_visit.size()) {
        vector <KDNode*> next_search;

        while(to_visit.size()) {
            cur = to_visit.back();
            to_visit.pop_back();

            int split_axis = cur->level % KDTREE_DIM;

            if(cur->left == NULL) {
                for(unsigned int i=0; i < cur->indexes.size(); i++) {
                    m_cmps++;

                    int idx = cur->indexes[i];
                    float d = Distance(query, pts[idx]);

                    if(d < best_dist) {
                        best_dist = d;
                        best_idx = idx;
                    }
                }
            }
            else {
                float d = query.coords[split_axis] - cur->split_value;

                // There are 3 possible scenarios
                // The hypercircle only intersects the left region
                // The hypercircle only intersects the right region
                // The hypercricle intersects both

                m_cmps++;

                if(fabs(d*d) > range) {
                    if(d < 0)
                        next_search.push_back(cur->left);
                    else
                        next_search.push_back(cur->right);
                }
                else {
                    next_search.push_back(cur->left);
                    next_search.push_back(cur->right);
                }
            }
        }

        to_visit = next_search;
    }

    *ret_index = best_idx;
    *ret_dist = best_dist;
}


void KDtree::SearchAtNodeRange(KDNode *cur, const Point &query, float range, vector<int> &ret_index, vector<float> &ret_sq_dist)
{
    vector <Point> &pts = *m_pts;
    vector <KDNode*> to_visit;

    to_visit.push_back(cur);

    while(to_visit.size()) {
        vector <KDNode*> next_search;

        while(to_visit.size()) {
            cur = to_visit.back();
            to_visit.pop_back();

            int split_axis = cur->level % KDTREE_DIM;

            if(cur->left == NULL) {
                for(unsigned int i=0; i < cur->indexes.size(); i++) {
                    m_cmps++;

                    int idx = cur->indexes[i];
                    float d = Distance(query, pts[idx]);

                    if(d < range) {
                        ret_sq_dist.push_back(d);
                        ret_index.push_back(idx);
                    }
                }
            }
            else {
                float d = query.coords[split_axis] - cur->split_value;

                // There are 3 possible scenarios
                // The hypercircle only intersects the left region
                // The hypercircle only intersects the right region
                // The hypercricle intersects both

                m_cmps++;

                if(fabs(d*d) > range) {
                    if(d < 0)
                        next_search.push_back(cur->left);
                    else
                        next_search.push_back(cur->right);
                }
                else {
                    next_search.push_back(cur->left);
                    next_search.push_back(cur->right);
                }
            }
        }

        to_visit = next_search;
    }

    //*ret_index = best_idx;
    //*ret_dist = best_dist;
}
*/

/*
// Comment by Guiming
// Search for the nearest neighbor
void KDtree::Search(const Point &query, int *ret_index, float *ret_dist)
{
    // Find the first closest node, this will be the upper bound for the next searches
    vector <Point> &pts = *m_pts;
    KDNode *best_node = NULL;
    int best_idx = 0;
    float best_dist = FLT_MAX;
    float radius = 0;
    m_cmps = 0;

    SearchAtNode(m_root, query, &best_idx, &best_dist, &best_node);

    radius = sqrt(best_dist);

    // Now find other possible candidates
    KDNode *cur = best_node;


    while(cur->parent != NULL) {
        // Go up
        KDNode *parent = cur->parent;
        int split_axis = (parent->level) % KDTREE_DIM;

        // Search the other node
        int tmp_idx;
        float tmp_dist = FLT_MAX;
        KDNode *tmp_node;
        KDNode *search_node = NULL;

        if(fabs(parent->split_value - query.coords[split_axis]) <= radius) {
            // Search opposite node
            if(parent->left != cur)
                SearchAtNodeRange(parent->left, query, radius, &tmp_idx, &tmp_dist);
            else
                SearchAtNodeRange(parent->right, query, radius, &tmp_idx, &tmp_dist);
        }

        if(tmp_dist < best_dist) {
            best_dist = tmp_dist;
            best_idx = tmp_idx;
        }

        cur = parent;
    }

    *ret_index = best_idx;
    *ret_dist = best_dist;
    //vector <Point> &pts = *m_pts
    //printf("%d %f %f %f\n", best_idx, best_dist, pts[best_idx].coords[0], pts[best_idx].coords[1]);
}
*/

// Added by Guiming
//Search for points within distance (squared) range
void KDtree::SearchRange(const Point &query, float range, vector<int> &ret_index, vector<float> &ret_sq_dist)
{
    //printf("== %f\n", sqrt(range));
    //SearchAtNodeRange(m_root, query, range, ret_index, ret_sq_dist);
    vector <Point> &pts = *m_pts;
    vector <KDNode*> to_visit;
    KDNode *cur = m_root;
    m_cmps = 0;
    to_visit.push_back(cur);

    while(to_visit.size()) {
        vector <KDNode*> next_search;

        while(to_visit.size()) {
            cur = to_visit.back();
            to_visit.pop_back();

            int split_axis = cur->level % KDTREE_DIM;

            if(cur->left == NULL) {
                //printf("cur->indexes.size(): %d\n", cur->indexes.size());
                  //int *indexes = &(cur->indexes[0]);
                for(unsigned int i=0; i < cur->indexes.size(); i++) {
                    m_cmps++;

                    //int idx = indexes[i];
                    int idx = cur->indexes[i];

                    float d = Distance(query, pts[idx]);

                    if(d < range) {
                        ret_sq_dist.push_back(d);
                        ret_index.push_back(idx);
                    }
                }
            }
            else {
                float d = query.coords[split_axis] - cur->split_value;

                // There are 3 possible scenarios
                // The hypercircle only intersects the left region
                // The hypercircle only intersects the right region
                // The hypercricle intersects both

                //m_cmps++;

                if(fabs(d*d) > range) {
                    if(d < 0)
                        next_search.push_back(cur->left);
                    else
                        next_search.push_back(cur->right);
                }
                else {
                    next_search.push_back(cur->left);
                    next_search.push_back(cur->right);
                }
            }
            //to_visit = next_search;
        }

        to_visit = next_search;
    }
    //printf("%d comparisons\n", m_cmps)  ;
}

/*
// Added by Guiming
// Brute-force to search for points within distance (squared) range
void KDtree::SearchRangeBruteForce(const Point &query, float range, vector<int> &ret_index, vector<float> &ret_sq_dist)
{
    //SearchAtNodeRange(m_root, query, range, ret_index, ret_sq_dist);
    vector <Point> &pts = *m_pts;
    for(unsigned int i=0; i < pts.size(); i++) {
        float d = Distance(query, pts[i]);
        if(d < range) {
            ret_sq_dist.push_back(d);
            ret_index.push_back(i);
        }
    }
}
*/
