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

#ifndef MOCK_MODELBOX_H_
#define MOCK_MODELBOX_H_

#include <modelbox/base/status.h>
#include <modelbox/flow.h>
#include <iostream>
#include <string>
#include "test_config.h"

namespace modelbox {

class MockModelBox {
 public:
  MockModelBox(){};
  virtual ~MockModelBox() { Stop(); };

  bool Init();
  void Stop();
  Status BuildAndRun(const std::string &name, const std::string &graph,
                     int timeout = 15 * 1000);
  std::shared_ptr<Flow> GetFlow();
  Status InitFlow(const std::string &name, const std::string &graph);
  std::vector<std::shared_ptr<BufferList>> GetOutputBufferList(
      std::shared_ptr<ExternalDataMap> ext_data, const std::string &port_name);

 private:
  std::shared_ptr<Flow> flow_;
};

}  // namespace modelbox
#endif  // MOCK_MODELBOX_H_
