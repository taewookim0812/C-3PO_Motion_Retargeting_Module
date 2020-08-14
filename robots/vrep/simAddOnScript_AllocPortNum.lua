function sysCall_init()
  portNum = tonumber(sim.getStringParameter(sim.stringparam_app_arg1))
  --print('RemoteAPI open by add on, PortNum:', portNum)

  if portNum > 0 then
    sim.setThreadSwitchTiming(2) -- Default timing for automatic thread switching
    -- (portNumber, maxPacketSize, debug, preEnableTrigger)
    simRemoteApi.start(portNum, 1500, false, true)
  end
end

function sysCall_cleanup()
  simRemoteApi.stop(portNum)
end

function sysCall_addOnScriptRun()
end
