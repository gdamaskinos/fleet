/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package utils;

import java.io.FileWriter;
import java.io.IOException;

import org.supercsv.io.CsvListWriter;
import org.supercsv.io.ICsvListWriter;
import org.supercsv.prefs.CsvPreference;

public class CsvWriter {
	
	ICsvListWriter csvWriter;
	
	public CsvWriter(String uri) throws IOException {
		csvWriter = new CsvListWriter(new FileWriter(uri), 
                CsvPreference.STANDARD_PREFERENCE);
	}
	

    public void writeCsv(String[][] csvMatrix) {

        try {
            for (int i = 0; i < csvMatrix.length; i++) {
                csvWriter.write(csvMatrix[i]);
            }
        } catch (IOException e) {
            e.printStackTrace(); // TODO handle exception properly
        } finally {
			try {
				csvWriter.flush();
			} catch (IOException e) {
				e.printStackTrace();
			}
		} 
    }
    
    public void writeCsv(double[][] csvMatrix) {
    	// parse double 2D matrix to String 2D matrix
    	String[][] output = new String[csvMatrix.length][csvMatrix[0].length];
		for (int row=0; row<csvMatrix.length; row++)
			for (int col=0; col<csvMatrix[0].length; col++)
				output[row][col] = String.valueOf(csvMatrix[row][col]);
		
        try {
            for (int i = 0; i < output.length; i++) {
                csvWriter.write(output[i]);
            }
        } catch (IOException e) {
            e.printStackTrace(); // TODO handle exception properly
        } finally {
			try {
				csvWriter.flush();
			} catch (IOException e) {
				e.printStackTrace();
			}
		} 
    }
    
    public void writeCsv(int[][] csvMatrix) {
    	// parse int 2D matrix to String 2D matrix
    	String[][] output = new String[csvMatrix.length][csvMatrix[0].length];
		for (int row=0; row<csvMatrix.length; row++)
			for (int col=0; col<csvMatrix[0].length; col++)
				output[row][col] = String.valueOf(csvMatrix[row][col]);
		
        try {
            for (int i = 0; i < output.length; i++) {
                csvWriter.write(output[i]);
            }
        } catch (IOException e) {
            e.printStackTrace(); // TODO handle exception properly
        } finally {
			try {
				csvWriter.flush();
			} catch (IOException e) {
				e.printStackTrace();
			}
		} 
    }
}
