/*
 * Copyright 2021 The Modelbox Project Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <signal.h>

#include "gtest/gtest.h"
#include "modelbox/base/log.h"
#include "modelbox/base/utils.h"

static int exit_signal;

static int g_sig_list[] = {
    SIGIO,   SIGPWR,    SIGSTKFLT, SIGPROF, SIGINT,  SIGTERM,
    SIGBUS,  SIGVTALRM, SIGTRAP,   SIGXCPU, SIGXFSZ, SIGILL,
    SIGABRT, SIGFPE,    SIGSEGV,   SIGQUIT, SIGSYS,
};
static int g_sig_num = sizeof(g_sig_list) / sizeof(g_sig_list[0]);
static constexpr int SIG_SKIP_STACK = 2;

static void test_sig_handler(int volatile sig_no, siginfo_t *sig_info,
                             void *volatile ptr) {
  switch (sig_no) {
    case SIGINT:
    case SIGTERM:
      exit_signal = sig_no;
      exit(1);
      break;
    case SIGQUIT:
      return;
      break;
    case SIGSEGV:
    case SIGPIPE:
    case SIGFPE:
    case SIGABRT:
    case SIGBUS:
    case SIGILL:
      std::cout << "Segment fault"
                << ", Signal: " << sig_no << ", Addr: " << sig_info->si_addr
                << ", Code: " << sig_info->si_code
                << ", Caused by: " << std::endl
                << modelbox::GetStackTrace(SIG_SKIP_STACK) << std::endl;
      usleep(300);
      break;
    default:
      break;
  }

  _exit(1);
}

static int test_sig_register() {
  int i = 0;
  struct sigaction sig_act;

  for (i = 0; i < g_sig_num; i++) {
    sig_act.sa_handler = nullptr;
    (void)sigemptyset(&sig_act.sa_mask);
    sig_act.sa_restorer = nullptr;
    sig_act.sa_sigaction = test_sig_handler;
    sig_act.sa_flags = SA_SIGINFO | SA_RESTART;

    if (sigaction(g_sig_list[i], &sig_act, nullptr) < 0) {
      fprintf(stderr, "Register signal %d failed.", g_sig_list[i]);
    }
  }

  return 0;
}

static int test_init() {
  if (test_sig_register() != 0) {
    fprintf(stderr, "register signal failed.\n");
    return 1;
  }

  return 0;
}

static void test_exit() {}

int main(int argc, char **argv) {
  int ret = 0;

  Defer { test_exit(); };
  /* Run init test */
  if (test_init() != 0) {
    fprintf(stderr, "init test failed.\n");
    return -1;
  }

  if (getenv("MODELBOX_CONSOLE_LOGLEVEL") == nullptr) {
    ModelBoxLogger.GetLogger()->SetLogLevel(modelbox::LOG_INFO);
  }

  ::testing::InitGoogleTest(&argc, argv);
  ret |= RUN_ALL_TESTS();

  return ret;
}
