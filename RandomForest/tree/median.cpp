#include "median.h"

namespace Yuki {
#define EPS 1e-6
	int WeightedPQueue::push(double data, double w) {
		if (size_ == value.size()) {
			value.push_back(data);
			weight.push_back(w);
		}
		else {
			value[size_] = data;
			weight[size_] = w;
		}
		++size_;
		
		// update
		int idx = size_ - 1;
		while (idx && value[idx - 1] > value[idx]) {
			{ double t = value[idx - 1];  value[idx - 1] = value[idx];   value[idx] = t; }
			{ double t = weight[idx - 1]; weight[idx - 1] = weight[idx]; weight[idx] = t; }
			--idx;
		}
		return 1;
	}

	int WeightedPQueue::remove(double data, double w) {
		int index = -1;
		for (size_t i = 0; i < size_; ++i) {
			if (abs(data - value[i]) < EPS && abs(w - weight[i]) < EPS) {
				index = i; break;
			}
		}
		if (index < 0) return 0;
		for (size_t i = index; i < size_ - 1; ++i) {
			value[i] = value[i + 1];
			weight[i] = weight[i + 1];
		}
		--size_;
		return 1;
	}

	int WeightedPQueue::pop(double &data, double &w) {
		if (size_) {
			data = value[0];
			w = weight[0];
			for (size_t i = 0; i < size_ - 1; ++i) {
				value[i] = value[i + 1];
				weight[i] = weight[i + 1];
			}
			--size_;
			return 1;
		}
		else {
			return 0;
		}
	}

	int WeightedPQueue::peek(double &data, double &w) {
		if (size_) {
			data = value[0];
			w = weight[0];
			return 1;
		}
		else {
			return 0;
		}
	}

	int WeightedMedianCalculator::push(double data, double w) {
		double original_median = 0;
		if (size() != 0)
			original_median = get_median();
		int ret = samples.push(data, w);
		update_median_parameters_post_push(data, w, original_median);
		return ret;
	}

	int WeightedMedianCalculator::remove(double data, double w) {
		double original_median = 0;
		if (size() != 0)
			original_median = get_median();
		int ret = samples.remove(data, w);
		update_median_parameters_post_remove(data, w, original_median);
		return ret;
	}

	int WeightedMedianCalculator::pop(double &data, double &w) {
		if (size() == 0) return 0;
		double original_median = get_median();
		int ret = samples.pop(data, w);
		update_median_parameters_post_remove(data, w, original_median);
		return ret;
	}

	int WeightedMedianCalculator::update_median_parameters_post_push(double data, double w, double original_median) {
		if (size() == 1) {
			k = 1;
			total_weight = sum_w_0_k = w;
			return 1;
		}
		total_weight += w;
		if (data < original_median) {
			++k;
			sum_w_0_k += w;

			while (k > 1 &&
				   (sum_w_0_k - samples.weight_at(k - 1) >= total_weight / 2.0)) {
				--k;
				sum_w_0_k -= samples.weight_at(k);
			}
		}
		else {
			while (k < samples.size() &&
				   (sum_w_0_k < total_weight / 2.0)) {
				sum_w_0_k += samples.weight_at(k);
				++k;
			}
		}
		return 1;
	}

	int WeightedMedianCalculator::update_median_parameters_post_remove(double data, double w, double original_median) {
		if (size() == 0) {
			k = 0;
			total_weight = sum_w_0_k = 0;
			return 1;
		}
		if (size() == 1) {
			k = 1;
			total_weight -= w;
			sum_w_0_k = total_weight;
			return 1;
		}
		total_weight -= w;
		if (data < original_median) {
			--k;
			sum_w_0_k -= w;
			
			while (k < samples.size() &&
				   (sum_w_0_k < total_weight / 2.0)) {
				sum_w_0_k += samples.weight_at(k);
				++k;
			}
		}
		else {
			while (k > 1 &&
				   (sum_w_0_k - samples.weight_at(k - 1) >= total_weight / 2.0)) {
				--k;
				sum_w_0_k -= samples.weight_at(k);
			}
		}
		return 1;
	}
}