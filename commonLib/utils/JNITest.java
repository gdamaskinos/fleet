/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package utils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class JNITest {

	public native void hello();

	static {
		String name = new File("./target/classes/libnative.so").getAbsolutePath();
		try {
			System.load(name);
		} catch (UnsatisfiedLinkError e2) {
			try {
				name = "native"; // EXECUTABLE=target/classes/libnative.so
				System.load(name);
			} catch (UnsatisfiedLinkError e) {
				try {
					String filename = System.mapLibraryName(name);
					InputStream in = JNITest.class.getClassLoader().getResourceAsStream(filename);
					int pos = filename.lastIndexOf('.');
					File file = File.createTempFile(filename.substring(0, pos), filename.substring(pos));
					file.deleteOnExit();
					try {
						byte[] buf = new byte[4096];
						OutputStream out = new FileOutputStream(file);
						try {
							while (in.available() > 0) {
								int len = in.read(buf);
								if (len >= 0) {
									out.write(buf, 0, len);
								}
							}
						} finally {
							out.close();
						}
					} finally {
						in.close();
					}
					System.load(file.getAbsolutePath());
				} catch (IOException e1) {
					// TODO Auto-generated catch block
					e1.printStackTrace();
				}
			}
		}
	}
}
