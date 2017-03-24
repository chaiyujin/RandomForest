#pragma once
#ifndef __YUKI_TREE_MEDIAN_H__
#define __YUKI_TREE_MEDIAN_H__

#include "../utils/log.h"
#include <vector>

namespace Yuki {
	class WeightedPQueue {
	public:
		WeightedPQueue(int init_size = 0)
			: size_(0),
			  value(init_size),
			  weight(init_size){}
		int push(double data, double w);
		int remove(double data, double w);
		int pop(double &data, double &w);
		int peek(double &data, double &w);
		double weight_at(int index) {
			CHECK(index < size());
			return weight[index];
		}
		double value_at(int index) {
			CHECK(index < size());
			return value[index];
		}
		size_t size() { return size_; }
		void reset() { size_ = 0; }
	private:
		int size_;
		std::vector<double> value;
		std::vector<double> weight;
	};

	class WeightedMedianCalculator {
	public:
		WeightedMedianCalculator(int init_size = 0)
			: samples(init_size), total_weight(0),
			  k(0), sum_w_0_k(0) {}
		~WeightedMedianCalculator() {}

		int push(double data, double w);
		int remove(double data, double w);
		int pop(double &data, double &w);
		double get_median() {
			if (sum_w_0_k == total_weight / 2.0) {
				return (samples.value_at(k) + 
					    samples.value_at(k - 1)) / 2.0;
			}
			else if (sum_w_0_k > total_weight / 2.0) {
				return samples.value_at(k - 1);
			}
			else {
				CHECK(0);
			}
		}
		

		void reset() {
			samples.reset();
			total_weight = 0;
			k = 0;
			sum_w_0_k = 0;
		}
		size_t size() { return samples.size(); }

	private:
		int update_median_parameters_post_push(double data, double weight, double original_median);
		int update_median_parameters_post_remove(double data, double weight, double original_median);

		WeightedPQueue samples;
		int k;
		double total_weight;
		double sum_w_0_k;
	};
}

#endif