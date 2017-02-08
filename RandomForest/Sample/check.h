#pragma once
#ifndef __CHECK_H__
#define __CHECK_H__

#include "../tree/param.h"
#include "../tree/feature_label.h"
#include "../tree/criterion.h"
#include "../tree/splitter.h"
#include "../utils/random.h"

#include <iostream>

void check_criterion();
void check_random();
void check_split();

#endif