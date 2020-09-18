#pragma once

#include <string>
#include <stdexcept>

namespace qutility {
	namespace history {
		namespace detail {
			template<typename T>
			[[nodiscard]] inline T* current(T* const& base_ptr, size_t const& single_size, size_t const& N_hist, size_t const& N_record) noexcept {
				return base_ptr + (N_record % N_hist) * single_size;
			}
			template<typename T>
			[[nodiscard]] inline const T* ccurrent(T* const& base_ptr, size_t const& single_size, size_t const& N_hist, size_t const& N_record) noexcept {
				return current(base_ptr, single_size, N_hist, N_record);
			}
			template<typename T>
			[[nodiscard]] inline T* former(T* const& base_ptr, size_t const& single_size, size_t const& N_hist, size_t const& N_Record) {
				if (N_Record == 0)
					throw std::invalid_argument("Cannot fetch the pointer to the former record of record No. 0.");
				return base_ptr + ((N_Record - 1) % N_hist) * single_size;
			}
			template<typename T>
			[[nodiscard]] inline const T* cformer(T* const& base_ptr, size_t const& single_size, size_t const& N_hist, size_t const& N_record) {
				return former(base_ptr, single_size, N_hist, N_record);
			}
			template<typename T>
			[[nodiscard]] inline T* latter(T* const& base_ptr, size_t const& single_size, size_t const& N_hist, size_t const& N_record) noexcept {
				return base_ptr + ((N_record + 1) % N_hist) * single_size;
			}
			template<typename T>
			[[nodiscard]] inline const T* clatter(T* const& base_ptr, size_t const& single_size, size_t const& N_hist, size_t const& N_record) noexcept {
				return latter(base_ptr, single_size, N_hist, N_record);
			}
			template<typename T>
			[[nodiscard]] inline T* at(T* const& base_ptr, size_t const& single_size, size_t const& N_hist, size_t const& N_record, ptrdiff_t const& pos) {
				if (pos >= (ptrdiff_t)N_hist)
					throw std::invalid_argument(
						std::string("At most ") + std::to_string(N_hist) + " record(s) allowed"
					);
				if (-pos > (ptrdiff_t)(N_hist > N_record ? N_record : N_hist))
					throw std::invalid_argument(
						std::string("Record ") + std::to_string(pos) + " requested while there are only "
						+ std::to_string(N_hist > N_record ? N_record : N_hist) + " record(s) avaliable."
					);
				return base_ptr + (((ptrdiff_t)N_record + pos) % N_hist) * single_size;
			}
			template<typename T>
			[[nodiscard]] inline const T* cat(T* const& base_ptr, size_t const& single_size, size_t const& N_hist, size_t const& N_record, ptrdiff_t const& pos) {
				return at(base_ptr, single_size, N_hist, N_record, pos);
			}
		}

		template <typename T>
		class HistoryBase {
		public:
			HistoryBase() = delete;
			~HistoryBase() = default;
			HistoryBase(const HistoryBase&) = delete;
			HistoryBase& operator= (const HistoryBase&) = delete;
			HistoryBase(HistoryBase&&) = default;
			HistoryBase& operator= (HistoryBase&&) = default;
			HistoryBase(T* const& base_ptr, size_t const& single_size, size_t const& N_hist)
				:base_ptr_(base_ptr), single_size_(single_size), N_hist_(N_hist) {}
			HistoryBase& push() noexcept {
				++N_record_;
				return *this;
			}
			HistoryBase& reset() noexcept {
				N_record_ = 0;
				return *this;
			}
			using type = T;
			[[nodiscard]] T* begin() const noexcept { return base_ptr_; };
			[[nodiscard]] const T* cbegin() const noexcept { return base_ptr_; };
			[[nodiscard]] auto available() const noexcept { return N_record_ > N_hist_ ? N_hist_ : N_record_; }
			[[nodiscard]] auto pos() const noexcept { return N_record_ % N_hist_; }
			[[nodiscard]] auto current() const noexcept { return detail::current(base_ptr_, single_size_, N_hist_, N_record_); }
			[[nodiscard]] auto ccurrent() const noexcept { return detail::ccurrent(base_ptr_, single_size_, N_hist_, N_record_); }
			[[nodiscard]] auto former() const { return detail::former(base_ptr_, single_size_, N_hist_, N_record_); }
			[[nodiscard]] auto cformer() const { return detail::cformer(base_ptr_, single_size_, N_hist_, N_record_); }
			[[nodiscard]] auto latter() const noexcept { return detail::latter(base_ptr_, single_size_, N_hist_, N_record_); }
			[[nodiscard]] auto clatter() const noexcept { return detail::clatter(base_ptr_, single_size_, N_hist_, N_record_); }
			[[nodiscard]] auto at(intptr_t const& pos) const { return detail::at(base_ptr_, single_size_, N_hist_, N_record_, pos); }
			[[nodiscard]] auto cat(intptr_t const& pos) const { return detail::cat(base_ptr_, single_size_, N_hist_, N_record_, pos); }

		protected:
			T* const base_ptr_;
			size_t const single_size_;
			size_t const N_hist_;
			size_t N_record_ = 0;
		};

		template<typename T, size_t SingleSize, size_t NHist>
		class History :public HistoryBase<T> {
		public:
			History() = delete;
			~History() = default;
			History(const History&) = delete;
			History& operator= (const History&) = delete;
			History(History&&) = default;
			History& operator= (History&&) = default;
			History(T* const& base_ptr) : HistoryBase<T>(base_ptr, SingleSize, NHist) {}
			constexpr static size_t single_size_ = SingleSize;
			constexpr static size_t N_hist_ = NHist;
		};

		template<typename T>
		class DHistory :public HistoryBase<T> {
		public:
			DHistory() = delete;
			~DHistory() = default;
			DHistory(const DHistory&) = delete;
			DHistory& operator= (const DHistory&) = delete;
			DHistory(DHistory&&) = default;
			DHistory& operator= (DHistory&&) = default;
			DHistory(T* const& base_ptr, size_t const& single_size, size_t const& N_hist) : HistoryBase<T>(base_ptr, single_size, N_hist) {}
		};
	}
}
