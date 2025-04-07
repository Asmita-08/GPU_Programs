#include <iostream>
#include <thread>

void print_hello(int id) {
    std::cout << "Hello from thread " << id << std::endl;
}

int main() {
    std::thread t1(print_hello, 1);
    std::thread t2(print_hello, 2);

    t1.join();
    t2.join();

    return 0;
}


