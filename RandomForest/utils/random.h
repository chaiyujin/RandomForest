/* -----------------------------
* File   : random.h
* Author : Yuki Chai
* Created: 2016.11.28
* Project: Yuki
*/
#pragma once
#ifndef __YUKI_RANDOM_H__
#define __YUKI_RANDOM_H__

#include "log.h"


#include <random>
using std::random_device;
using std::seed_seq;
using std::mt19937;
using std::uniform_real_distribution;

namespace Yuki {
    class Random {
    private:
        random_device   r;
        seed_seq        seed;
        mt19937         engine;
        uniform_real_distribution<> dist;
    public:
        Random() : r(), 
            seed({r(), r(), r(), r(), r()}),
            engine(seed),
            dist(0, 1) {}
        ~Random() {}

        float random() {
            return dist(engine);
        }

        template <class T>
        T random(T scale) {
            return random() * scale;
        }
    };
}

#endif // !__YUKI_RANDOM_H__
