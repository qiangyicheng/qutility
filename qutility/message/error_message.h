#pragma once

#include<iostream>

namespace qutility {
	namespace message {
		inline void exit_with_message(char const* const message, char const* const file, int const line) {
			std::cerr << "Error at " << file << ":" << line << ":\n"
				<< message << std::endl;
			exit(1);
		}
	}
}