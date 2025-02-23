# ISACPP
- `pisDL.go`: contain the plugin implementation of ISACPP
- `server.py`: receive the configuration of the incoming workload and return the scheduling decision based on ISACPP scheduling algorithm

Note that ISACPP is required to registered as a scheduling plugin of Volcano. The detailed plugin registration process please refer to [here](https://github.com/volcano-sh/volcano/tree/master/example/custom-plugin)

When registering the plugin, remember to replace the remote volcano library with the local one. More details can be found [here](https://github.com/volcano-sh/volcano/issues/3174)
