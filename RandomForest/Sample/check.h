#pragma once
#ifndef __CHECK_H__
#define __CHECK_H__

#include "../tree/param.h"
#include "../tree/feature_label.h"
#include "../tree/criterion.h"
#include "../tree/splitter.h"
#include "../tree/tree.h"
#include "../forest/random_forest.h"
#include "../utils/random.h"
#include "../utils/timer.h"

#include <iostream>

void check_criterion();
void check_random();
void check_split();
void check_tree_1();
void check_tree_2();

void check_foreset_1();

void check_load();
void check_load_forest();

#endif