CUSTOM_COMM_ROOT ?= ..
ASCEND_CANN_PACKAGE_PATH ?= $(ASCEND_HOME_PATH)
ifeq ($(ASCEND_CANN_PACKAGE_PATH),)
ASCEND_CANN_PACKAGE_PATH := /usr/local/Ascend/ascend-toolkit/latest
endif

TARGET := custom_comm_allgather_batch_testcase
CUSTOM_COMM_LIB_DIR = $(abspath $(CUSTOM_COMM_ROOT)/python/custom_comm)

CXX ?= g++
CXXFLAGS ?= -O2 -g
CXXFLAGS += -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 \
            -DHCCL_COMM_HCCL_QOS_CONFIG_NOT_SET=HCCL_COMM_QOS_CONFIG_NOT_SET \
            -Wall -Wextra -Wno-unused-parameter
CPPFLAGS += -I$(CUSTOM_COMM_ROOT)/ops/allgather_batch/inc \
            -I$(ASCEND_CANN_PACKAGE_PATH)/include \
            -I$(ASCEND_CANN_PACKAGE_PATH)/include/hccl \
            -I$(ASCEND_CANN_PACKAGE_PATH)/pkg_inc \
            -I$(ASCEND_CANN_PACKAGE_PATH)/pkg_inc/hcomm \
            -I$(ASCEND_CANN_PACKAGE_PATH)/pkg_inc/hcomm/ccu \
            -I$(ASCEND_CANN_PACKAGE_PATH)/hcomm/hcomm/include \
            -I$(ASCEND_CANN_PACKAGE_PATH)/hcomm/hcomm/include/hccl \
            -I$(ASCEND_CANN_PACKAGE_PATH)/hcomm/hcomm/pkg_inc \
            -I$(ASCEND_CANN_PACKAGE_PATH)/hcomm/hcomm/pkg_inc/hcomm \
            -I$(ASCEND_CANN_PACKAGE_PATH)/hcomm/hcomm/pkg_inc/hcomm/ccu \
            -I$(ASCEND_CANN_PACKAGE_PATH)/npu-runtime/runtime/include/external
LDFLAGS += -L$(CUSTOM_COMM_LIB_DIR) \
           -L$(ASCEND_CANN_PACKAGE_PATH)/lib64 \
           -L$(ASCEND_CANN_PACKAGE_PATH)/x86_64-linux/lib64 \
           -L$(ASCEND_CANN_PACKAGE_PATH)/hcomm/hcomm/lib64 \
           -Wl,-rpath,$(CUSTOM_COMM_LIB_DIR) \
           -Wl,-rpath,$(ASCEND_CANN_PACKAGE_PATH)/lib64 \
           -Wl,-rpath,$(ASCEND_CANN_PACKAGE_PATH)/x86_64-linux/lib64 \
           -Wl,-rpath,$(ASCEND_CANN_PACKAGE_PATH)/hcomm/hcomm/lib64
LDLIBS += -lcustom_comm_impl -lhccl -lhcomm -lascendcl -lpthread

.PHONY: all check-lib clean run

all: check-lib $(TARGET)

check-lib:
	@test -f "$(CUSTOM_COMM_LIB_DIR)/libcustom_comm_impl.so" || \
		(echo "missing $(CUSTOM_COMM_LIB_DIR)/libcustom_comm_impl.so"; \
		 echo "please build custom_comm first, e.g. cd .. && python3 setup.py build_ext --inplace"; \
		 exit 1)

$(TARGET): main.cc
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

run: $(TARGET)
	LD_LIBRARY_PATH="$(CUSTOM_COMM_LIB_DIR):$(ASCEND_CANN_PACKAGE_PATH)/lib64:$(ASCEND_CANN_PACKAGE_PATH)/x86_64-linux/lib64:$(ASCEND_CANN_PACKAGE_PATH)/hcomm/hcomm/lib64:$$LD_LIBRARY_PATH" ./$(TARGET)

clean:
	rm -f $(TARGET)
