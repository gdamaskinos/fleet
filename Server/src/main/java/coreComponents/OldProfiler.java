/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package coreComponents;

import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.Map;


/**
 * Deprecated thread-task profiler
 * Abstract class for implementing a profiler
 * A profiler is a thread-task that spawns other thread-tasks for handling client requests
 * @author damaskin
 *
 */
public abstract class OldProfiler implements Runnable {

	final ServerSocket profilerSocket;
	
	abstract void handleClientRequest(Socket socket);
	
	public OldProfiler(ServerSocket profilerSocket) {
		this.profilerSocket = profilerSocket;
	}
	
	@Override
	/**
	 * Main thread-task running
	 */
	public void run() {
		try {
			while (true) {
				Socket socket = profilerSocket.accept();
				
				// when a request arrives, run a handler thread to handle it 
				HandleClient handleTask = new HandleClient(socket);

				//HandleClient handleTask = new HandleClient(socket, model);
				new Thread(handleTask).start();
			}
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				profilerSocket.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
	
    class HandleClient implements Runnable {

		private Socket socket;
		
		public HandleClient(Socket socket){
			this.socket = socket;
		}
         
		public void run() {
			handleClientRequest(socket);
		}
	}
    
    
}
