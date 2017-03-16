#pragma once
#ifndef __TREE_FEATURE_LABEL_H__
#define __TREE_FEATURE_LABEL_H__

#include "../utils/log.h"
#include "param.h"
#include <vector>
#include <algorithm>

/* Feature and label should change according to data */
namespace Yuki {

	template <class T>
	class Label {
	public:
		Label() {}

		size_t read(FILE *fp, int size) {
			v.resize(size);
			return fread(v.data(), sizeof(T), size, fp);
		}
		Label &operator+=(const Label &l) {
			CHECK(v.size() == l.v.size());
			for (size_t i = 0; i < v.size(); ++i) {
				v[i] += l.v[i];
			}
			return *this;
		}
		Label &operator*=(float f) {
			for (size_t i = 0; i < v.size(); ++i) {
				v[i] *= f;
			}
			return *this;
		}
		bool operator==(const Label &l) {
			if (v.size() != l.v.size()) return false;
			for (size_t i = 0; i < v.size(); ++i) {
				if (!equal(v[i], l.v[i])) return false;
			}
			return true;
		}
		T &operator[](int index) {
			CHECK(0 <= index && index < v.size());
			return v[index];
		}
		const T &operator[](int index) const {
			CHECK(0 <= index && index < v.size());
			return v[index];
		}
		size_t size() const { return v.size(); }
		void resize(int size) { v.resize(size); }

		T *data() { return v.data(); }
		const T *data() const { return v.data(); }

		void zeros(int size) {
			v.resize(size);
			memset(v.data(), 0, sizeof(T) * size);
		}
	private:
		std::vector<T> v;
	};

	template <class T>
	class Feature {
	public:
		Feature() {}

		size_t read(FILE *fp, int size) {
			v.resize(size);
			return fread(v.data(), sizeof(T), size, fp);
		}
		T &operator[](int index) {
			CHECK(0 <= index && index < v.size());
			return v[index];
		}
		const T &operator[](int index) const {
			CHECK(0 <= index && index < v.size());
			return v[index];
		}

		bool operator==(const Feature &f) {
			for (size_t i = 0; i < v.size(); ++i) {
				if (!equal(v[i], f.v[i])) return false;
			}
			return true;
		}

		T *data() { return v.data(); }
		const T *data() const { return v.data(); }

		size_t size() const { return v.size(); }
		void resize(int size) { v.resize(size); }

		void zeros(int size) {
			v.resize(size);
			memset(v.data(), 0, sizeof(T) * size);
		}

		template <class U>
		friend bool less_than(const Feature<U> &f0, const Feature<U> &f1, int dim, const Mask &mask);
	private:
		std::vector<T> v;
	};

	/* the sort is designed for the feature */
	template <class T>
	inline bool less_than(const Feature<T> &f0, const Feature<T> &f1, int dim, const Mask &mask) {
#define CMP_FEATURE(i) \
		if (mask[i]) {if (f0[i] < f1[i]) return true;\
		else if (f0[i] > f1[i]) return false;}
	
		CMP_FEATURE(dim);

		int delta = std::max(dim, (int)f0.size() - dim - 1);
		for (int i = 1; i <= delta; ++i) {
			int j = dim - i;
			if (j >= 0) CMP_FEATURE(j);
			j = dim + i;
			if (j < f0.size()) CMP_FEATURE(j);
		}
#undef CMP_FEATURE
		return false;
	}

	template <class T>
	inline bool less_than(const Feature<T> &f0, const Feature<T> &f1, int dim, const SetMask &set_split, const Mask &mask) {
#define CMP_FEATURE(i) \
		if (mask[i]) {if (set_split[f0[i]] < set_split[f1[i]]) return true;\
		else if (set_split[f0[i]] > set_split[f1[i]]) return false;}

		CMP_FEATURE(dim);

		int delta = std::max(dim, (int)f0.size() - dim - 1);
		for (int i = 1; i <= delta; ++i) {
			int j = dim - i;
			if (j >= 0) CMP_FEATURE(j);
			j = dim + i;
			if (j < f0.size()) CMP_FEATURE(j);
		}
#undef CMP_FEATURE
		return false;
	}

	/* define the data feature and label */
	typedef Feature<int> DFeature;
	typedef Label<float> DLabel;

	class Tuple {
	public:
		// basic data
		DFeature X;
		DLabel   Y;
		
		// auxillary info
		size_t  id;
		float   weight; // default as 1.f

		Tuple() : weight(1.f) {}
		
		// no copy, which is a waste of time
		Tuple(const Tuple &) = delete;
		Tuple &operator=(const Tuple &) = delete;
	};

	// sort by feature id ( for numeric feature )
	class TupleSorter {
	public:
		TupleSorter(int i, const Mask &mask) : dim(i), mask(mask) {}
		bool operator()(Tuple * const &t0, Tuple * const &t1) {
			return less_than(t0->X, t1->X, dim, mask);
		}
	private:
		int dim;
		const Mask &mask;
	};

	// sort by set (split categoric feature into several sets)
	class SetSorter {
	public:
		SetSorter(int i, const SetMask &set_split, const Mask &mask)
			: dim(i), set_split(set_split), mask(mask) {}
		bool operator()(Tuple *const &t0, Tuple *const &t1) {
			return less_than(t0->X, t1->X, dim, set_split, mask);
		}
	private:
		int dim;
		const SetMask set_split;
		const Mask mask;
	};

	typedef std::vector<Tuple *> DataSet;

	DataSet read_data(
		const char *feat_file_name,
		const char *label_file_name,
		const Param &param);
	DataSet read_data(
		const char *feat_file_name,
		const Param &param);
	
}

#endif  // !__TREE_FEATURE_LABEL_H__