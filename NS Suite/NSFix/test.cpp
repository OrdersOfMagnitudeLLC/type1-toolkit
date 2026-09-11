// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#include "nsfix.hpp"
#include <iostream>

using namespace nsfix;

void on_app_message(Message& msg) {
    std::cout << "Application message received!" << std::endl;
    std::cout << "  MsgType: " << msg.get(Tag::MsgType) << std::endl;
    std::cout << "  Symbol: " << msg.get(Tag::Symbol) << std::endl;
    std::cout << "  Side: " << msg.get(Tag::Side) << std::endl;
    std::cout << "  OrderQty: " << msg.get_int(Tag::OrderQty) << std::endl;
    std::cout << "  Price: " << msg.get_double(Tag::Price) << std::endl;
    std::cout << "  ClOrdID: " << msg.get(Tag::ClOrdID) << std::endl;
}

int main() {
    std::cout << "=== NSFix Test ===" << std::endl;
    
    // Create session with callback
    Session session("CLIENT", "BROKER", on_app_message);
    
    // Test 1: Send logon
    std::cout << "\n1. Sending logon..." << std::endl;
    std::string_view logon_msg = session.logon();
    std::cout << "  Logon message: " << logon_msg << std::endl;
    std::cout << "  State: " << (int)session.state() << " (should be 1=LOGON_SENT)" << std::endl;
    
    // Test 2: Receive logon acknowledgment from server
    std::cout << "\n2. Receiving logon acknowledgment..." << std::endl;
    const char* logon_ack = "8=FIX.4.2\x01"
                            "9=70\x01"
                            "35=A\x01"
                            "49=BROKER\x01"
                            "56=CLIENT\x01"
                            "34=1\x01"
                            "52=20240101-12:00:00\x01"
                            "98=0\x01"
                            "108=30\x01"
                            "10=123\x01";
    session.feed(logon_ack, strlen(logon_ack));
    std::cout << "  State: " << (int)session.state() << " (should be 2=ACTIVE)" << std::endl;
    std::cout << "  SeqIn: " << session.seq_in() << " (should be 2)" << std::endl;
    
    // Test 3: Receive NewOrderSingle (application message)
    std::cout << "\n3. Receiving NewOrderSingle..." << std::endl;
    const char* new_order = "8=FIX.4.2\x01"
                            "9=120\x01"
                            "35=D\x01"
                            "49=BROKER\x01"
                            "56=CLIENT\x01"
                            "34=2\x01"
                            "52=20240101-12:00:01\x01"
                            "11=ORD12345\x01"
                            "55=AAPL\x01"
                            "54=1\x01"
                            "38=100\x01"
                            "40=2\x01"
                            "44=150.50\x01"
                            "10=200\x01";
    session.feed(new_order, strlen(new_order));
    
    // Test 4: Send logout
    std::cout << "\n4. Sending logout..." << std::endl;
    std::string_view logout_msg = session.logout();
    std::cout << "  Logout message: " << logout_msg << std::endl;
    std::cout << "  State: " << (int)session.state() << " (should be 3=LOGOUT_SENT)" << std::endl;
    
    std::cout << "\n=== Test Complete ===" << std::endl;
    
    return 0;
}
