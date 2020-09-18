#pragma once

#include <fstream>
#include <iomanip>
#include <string>
#include <type_traits>

namespace qutility {
	namespace matio {
		//templates for compatibility with restrict keyword
		template < typename T > struct remove_restrict { typedef T type; };
		template < typename T > struct remove_restrict<T*> { typedef T* type; };
		template < typename T > struct remove_restrict<T* __restrict> { typedef T* type; };

		//two different requiements
		template <typename P>
		struct is_my_pointer {
			constexpr static bool value = std::is_pointer<typename remove_restrict<P>::type>::value;
		};

		template <typename S>
		struct is_my_size {
			constexpr static bool value = std::is_same<S, size_t>::value;
		};

		//constrains for std::pair with two different requirements
		template < typename P, typename S,
			typename = typename std::enable_if<
			is_my_pointer<P>::value&&
			is_my_size<S>::value
			, void>::type
		>
			using MyPair = std::pair<P, S>;

		//templates for friendly compile message

		template <typename... Arg>
		struct is_pointer_size_alternating;

		template<typename U, typename... Arg>
		struct is_pointer_size_alternating<U, Arg...> {
			constexpr static bool value =
				(
					(
						is_my_pointer<U>::value &&
						((sizeof...(Arg)) % 2 == 1)
						) ||
					(
						is_my_size<U>::value &&
						((sizeof...(Arg)) % 2 == 0)
						)
					) &&
				is_pointer_size_alternating<Arg...>::value;
		};

		template <typename U>
		struct is_pointer_size_alternating<U> {
			constexpr static bool value =
				is_my_size<U>::value;
		};

		template <typename... Arg>
		struct is_pointer_size_list;

		template <typename U, typename... Arg>
		struct is_pointer_size_list<U, Arg...> {
			constexpr static bool value =
				((sizeof...(Arg)) % 2) == 1 &&
				is_pointer_size_alternating<U, Arg...>::value;
		};

		template <typename U>
		struct is_pointer_size_list<U> {
			constexpr static bool value = false;
		};


		template <typename... Arg>
		struct is_pointer_list;

		template <typename U, typename... Arg>
		struct is_pointer_list<U, Arg...> {
			constexpr static bool value =
				is_my_pointer<U>::value &&
				is_pointer_list<Arg...>::value;
		};

		template <typename U>
		struct is_pointer_list<U> {
			constexpr static bool value =
				is_my_pointer<U>::value;
		};

		//basic impl for input, using ifs

		template< typename T>
		void ReadMatrix(std::ifstream& ifs, MyPair<T, size_t> First) {
			std::cout << "REPORT: reading " << First.second << " elements to 0x" << std::hex << First.first << std::dec << std::endl;
			ifs.read((char*)First.first, sizeof(typename std::remove_pointer<typename remove_restrict<T>::type>::type) * First.second);
			if (ifs.rdstate() != std::ios_base::goodbit) {
				std::cout << "ERROR: ifstream error when reading " << First.second << " elements to 0x" << std::hex << First.first << std::dec << std::endl;
				exit(1);
			}
			return;
		}

		template< typename T, typename... Arg >
		void ReadMatrix(std::ifstream& ifs, MyPair<T, size_t> First, MyPair<Arg, size_t>... Rest) {
			std::cout << "REPORT: reading " << First.second << " elements to 0x" << std::hex << First.first << std::dec << std::endl;
			ifs.read((char*)First.first, sizeof(typename std::remove_pointer<typename remove_restrict<T>::type>::type) * First.second);
			if (ifs.rdstate() != std::ios_base::goodbit) {
				std::cout << "ERROR: ifstream error when reading " << First.second << " elements to 0x" << std::hex << First.first << std::dec << std::endl;
				exit(1);
			}
			ReadMatrix(ifs, Rest...);
			return;
		}

		//basic impl for output, using ofs

		template< typename T>
		void WriteMatrix(std::ofstream& ofs, MyPair<T, size_t> First) {
			ofs.write((char*)First.first, sizeof(typename std::remove_pointer<typename remove_restrict<T>::type>::type) * First.second);
			if (ofs.rdstate() != std::ios_base::goodbit) {
				std::cout << "ERROR: ofstream error when writing " << First.second << " elements from 0x" << std::hex << First.first << std::dec << std::endl;
				exit(1);
			}
			return;
		}

		template< typename T, typename... Arg >
		void WriteMatrix(std::ofstream& ofs, MyPair<T, size_t> First, MyPair<Arg, size_t>... Rest) {
			ofs.write((char*)First.first, sizeof(typename std::remove_pointer<typename remove_restrict<T>::type>::type) * First.second);
			if (ofs.rdstate() != std::ios_base::goodbit) {
				std::cout << "ERROR: ofstream error when writing " << First.second << " elements from 0x" << std::hex << First.first << std::dec << std::endl;
				exit(1);
			}
			WriteMatrix(ofs, Rest...);
			return;
		}

		//sugars for input, using ifs

		template <typename U, typename V,
			typename = typename std::enable_if<
			is_pointer_size_list<U, V>::value
			, void>::type
		>
			void ReadMatrix(std::ifstream& ifs, U ptr, V size) {
			ReadMatrix(ifs, std::make_pair(ptr, size));
			return;
		}

		template <typename U, typename V, typename... Arg,
			typename = typename std::enable_if<
			is_pointer_size_list<U, V, Arg...>::value
			, void>::type
		>
			void ReadMatrix(std::ifstream& ifs, U ptr, V size, Arg... Rest) {
			ReadMatrix(ifs, std::make_pair(ptr, size));
			ReadMatrix(ifs, Rest...);
			return;
		}

		template <typename... Arg,
			typename = typename std::enable_if<
			is_pointer_list<Arg...>::value
			, void>::type
		>
			void ReadMatrix(std::ifstream& ifs, size_t size, Arg... Rest) {
			ReadMatrix(ifs, std::make_pair(Rest, size)...);
			return;
		}

		//sugars for output, using ofs

		template <typename U, typename V,
			typename = typename std::enable_if<
			is_pointer_size_list<U, V>::value
			, void>::type
		>
			void WriteMatrix(std::ofstream& ofs, U ptr, V size) {
			WriteMatrix(ofs, std::make_pair(ptr, size));
			return;
		}

		template <typename U, typename V, typename... Arg,
			typename = typename std::enable_if<
			is_pointer_size_list<U, V, Arg...>::value
			, void>::type
		>
			void WriteMatrix(std::ofstream& ofs, U ptr, V size, Arg... Rest) {
			WriteMatrix(ofs, std::make_pair(ptr, size));
			WriteMatrix(ofs, Rest...);
			return;
		}

		template <typename... Arg,
			typename = typename std::enable_if<
			is_pointer_list<Arg...>::value
			, void>::type
		>
			void WriteMatrix(std::ofstream& ofs, size_t size, Arg... Rest) {
			WriteMatrix(ofs, std::make_pair(Rest, size)...);
			return;
		}

		//api for input using filename

		template< typename... Arg >
		void ReadMatrix(std::string Filename, MyPair<Arg, size_t>... Rest) {
			std::ifstream ifs;
			ifs.open(Filename, std::ios::in | std::ios::binary);
			ReadMatrix(ifs, Rest...);
			ifs.close();
			return;
		}

		template <typename... Arg,
			typename = typename std::enable_if<
			is_pointer_size_list<Arg...>::value
			, void>::type
		>
			void ReadMatrix(std::string Filename, Arg... Rest) {
			std::ifstream ifs;
			ifs.open(Filename, std::ios::in | std::ios::binary);
			ReadMatrix(ifs, Rest...);
			ifs.close();
			return;
		}

		template <typename... Arg,
			typename = typename std::enable_if<
			is_pointer_list<Arg...>::value
			, void>::type
		>
			void ReadMatrix(std::string Filename, size_t size, Arg... Rest) {
			std::ifstream ifs;
			ifs.open(Filename, std::ios::in | std::ios::binary);
			ReadMatrix(ifs, size, Rest...);
			ifs.close();
			return;
		}

		//api for output using filename

		template< typename... Arg >
		void WriteMatrix(std::string Filename, MyPair<Arg, size_t>... Rest) {
			std::ofstream ofs;
			ofs.open(Filename, std::ios::out | std::ios::binary);
			WriteMatrix(ofs, Rest...);
			ofs.close();
			return;
		}

		template <typename... Arg,
			typename = typename std::enable_if<
			is_pointer_size_list<Arg...>::value
			, void>::type
		>
			void WriteMatrix(std::string Filename, Arg... Rest) {
			std::ofstream ofs;
			ofs.open(Filename, std::ios::out | std::ios::binary);
			WriteMatrix(ofs, Rest...);
			ofs.close();
			return;
		}

		template <typename... Arg,
			typename = typename std::enable_if<
			is_pointer_list<Arg...>::value
			, void>::type
		>
			void WriteMatrix(std::string Filename, size_t size, Arg... Rest) {
			std::ofstream ofs;
			ofs.open(Filename, std::ios::out | std::ios::binary);
			WriteMatrix(ofs, size, Rest...);
			ofs.close();
			return;
		}

	}
}