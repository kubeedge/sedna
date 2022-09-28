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

#include "mock_modelbox.h"
#include <sstream>
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "modelbox/data_context.h"
#include "modelbox/session_context.h"

using ::testing::_;
namespace modelbox {

Status MockModelBox::InitFlow(const std::string &name,
                              const std::string &graph) {
  flow_ = std::make_shared<Flow>();
  return flow_->Init(name, graph);
}

Status MockModelBox::BuildAndRun(const std::string &name,
                                 const std::string &graph, int timeout) {
  auto ret = InitFlow(name, graph);
  if (!ret) {
    return ret;
  }

  ret = flow_->Build();
  if (!ret) {
    return ret;
  }

  ret = flow_->RunAsync();
  if (!ret) {
    return ret;
  }

  if (timeout < 0) {
    return ret;
  }

  Status retval;
  flow_->Wait(timeout, &retval);
  return retval;
}

void MockModelBox::Stop() {
  if (flow_ != nullptr) {
    flow_->Stop();
    flow_ = nullptr;
  }
}

std::shared_ptr<Flow> MockModelBox::GetFlow() { return flow_; }

std::vector<std::shared_ptr<BufferList>> MockModelBox::GetOutputBufferList(
    std::shared_ptr<ExternalDataMap> ext_data, const std::string &port_name) {
  Status status;
  std::vector<std::shared_ptr<BufferList>> output_buffer_lists;
  while (true) {
    OutputBufferList map_buffer_list;
    status = ext_data->Recv(map_buffer_list);
    if (status == STATUS_SUCCESS) {
      auto buffer_list = map_buffer_list[port_name];
      output_buffer_lists.push_back(buffer_list);
    } else {
      EXPECT_EQ(status, STATUS_EOF);
      break;
    }
  }
  return output_buffer_lists;
}

}  // namespace modelbox
