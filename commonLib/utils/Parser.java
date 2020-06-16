/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package utils;

/**
 * Parses target values for MINST dataset.
 * csv file row needs to be in the format: [target_value]
 */
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;

import org.jblas.DoubleMatrix;

import au.com.bytecode.opencsv.CSVReader;

/**
 * Parser for the examples' values
 */
public class Parser {

	File f;
	CSVReader reader;
	
	/**
	 * Number of examples (rows) in csv input file
	 */
	public int rows;
	
	/**
	 * Number of labels/input features (cols) in csv input file
	 */
	public int cols;
	
	
	/**
	 * Constructs a parser for a csv file
	 * The file must be in the format (#examples X #labels) or (#examples X #features)
	 * @param uri of input csv file
	 * @throws IOException
	 */
	public Parser(String uri) throws IOException {
		f = new File(uri);
		FileReader fr = new FileReader(f);
		reader = new CSVReader(fr);
		rows = this.rows();
		cols = this.cols();
	}

	private void reset() throws IOException {
		reader.close();
		reader = new CSVReader(new FileReader(f));
	}

	private void reset(int line) throws IOException {
		reader.close();
		reader = new CSVReader(new FileReader(f), ',', '\'', line);
	}
	
	private int rows() throws IOException {
		LineNumberReader lreader = new LineNumberReader(new FileReader(f));
		int count = 0;
		while (lreader.readLine() != null) 
			count++;
		lreader.close();
		return count;
		//return reader.readAll().size();
	}

	private int cols() throws IOException {
		reset();
		return reader.readNext().length;
	}
	
	/**
	 * Reads a single line from input file without loading it to memory.
	 * @param line
	 * @return DoubleMatrix: 1 x {@link Parser#cols}
	 * @throws IOException
	 */
	public DoubleMatrix getSingleRow(int line) throws IOException {
		reset(line);
		DoubleMatrix res = new DoubleMatrix(1, cols);
		int j = 0;
		for (String value : reader.readNext()) {
			res.put(j, Double.parseDouble(value));
			j++;
		}
		return res;
	}

	/**
	 * Parse multiple columns csv file.
	 * e.g. rowX: [0, 0, 1, 0, 0, 1, ...] -> labels
	 * e.g. rowX: [0.0301, 0.202, ...] -> input features
	 * @return DoubleMatrix: {@link Parser#rows} x {@link Parser#cols}
	 */
	public DoubleMatrix getValueMatrix() throws IOException {	
		DoubleMatrix res = new DoubleMatrix(rows, cols);
		reset();
		int i=0, j;
		while (i < rows) {
			j = 0;
			for (String value : reader.readNext()) {
				res.put(i, j, Double.parseDouble(value));
				j++;
			}
			i++;
		}
		return res;
	}
	
	/**
	 * Parse multiple columns csv file.
	 * e.g. rowX: [0, 0, 1, 0, 0, 1, ...] -> labels
	 * e.g. rowX: [0.0301, 0.202, ...] -> input features
	 * @return double[][]: {@link Parser#rows} x {@link Parser#cols}
	 */
	public double[][] getValues() throws IOException {	
		double[][] res = new double[rows][cols];
		reset();
		int i=0, j;
		while (i < rows) {
			j = 0;
			for (String value : reader.readNext()) {
				res[i][j] = Double.parseDouble(value);
				j++;
			}
			i++;
		}
		return res;
	}
	
}
