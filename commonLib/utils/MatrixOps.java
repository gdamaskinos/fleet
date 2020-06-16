/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package utils;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import org.jblas.DoubleMatrix;

public class MatrixOps {
	/**
	 * Get subarray. Pay attention to index ranges following MATLAB's notation.
	 * 
	 * @param x
	 *            : DoubleMatrix ( M x N )
	 * @param row
	 *            : 0 <= row < M (normal row pointer)
	 * @param start
	 *            : 1 <= start < N (matlab style)
	 * @param end
	 *            : start < end < N (matlab style)
	 * @param step
	 * @return A single row-matrix.
	 */
	public static DoubleMatrix getRowRange(DoubleMatrix x, int row, int start, int step, int end) {
		int size = 0; // output size
		for (int i = start - 1; i < end; i += step)
			size++;
		DoubleMatrix res = new DoubleMatrix(1, size);
		
		int j = 0; // output index
		for (int i = start - 1; i < end; i += step, j++) {
			res.put(j, x.get(row, i));
		}
		return res;
	}

	/**
	 * Maps g function: g(X, Y) = X./(1+exp(-Y)) to an DoubleMatrix.
	 * 
	 * @param X
	 *            : Matrix ( N x M )
	 * @param Y
	 *            : Matrix ( N x M )
	 * @return Matrix ( N x M )
	 */
	public static DoubleMatrix gFunction(DoubleMatrix X, DoubleMatrix Y) {
		int rows = X.rows;
		int cols = X.columns;

		DoubleMatrix res = new DoubleMatrix(rows, cols);
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				res.put(i, j, X.get(i, j) / (1 + Math.exp(-Y.get(i, j))));
			}
		}
		return res;
	}

	public static DoubleMatrix readMatrix(BufferedReader reader) {
		try {
			int i, j;
			
			int dataLen = Integer.parseInt(reader.readLine());
			char[] data = new char[dataLen];
			
			reader.read(data, 0, dataLen);
			//String compressedStr = new String(data);
			String matrix_str = new String(data);
			
			//String matrix_str = decompress(compressedStr);
			String[] matrixData = matrix_str.split(",");
			
			int m = Integer.parseInt(matrixData[0].trim());
			int n = Integer.parseInt(matrixData[1].trim());
			
			DoubleMatrix mat = new DoubleMatrix(m, n);
			for (i = 0; i < m; i++) {
				for (j = 0; j < n; j++) {

					double m_ij = Double.parseDouble(matrixData[n * i + j+2].trim());
					mat.put(i, j, m_ij);
				}
			}

			return mat;
		} catch (IOException e) {

			e.printStackTrace();
			return null;
		}
	}

	public static void printMatrix(DoubleMatrix mat, PrintWriter printWriter) {
		int rows = mat.rows;
		int cols = mat.columns;

		String matrix_str = rows+","+cols+",";
		
		double[][] mArray = mat.toArray2();
		for (int i = 0; i < rows - 1; i++) {
			String rowStr = Arrays.toString(mArray[i]);
			rowStr = rowStr.substring(1, rowStr.length() - 1) + ",";
			matrix_str += rowStr;
		}

		String lastRow = Arrays.toString(mArray[rows - 1]);
		
		lastRow = lastRow.substring(1, lastRow.length() - 1);
		matrix_str += lastRow;
		
		String compressedStr = compress(matrix_str);
		compressedStr = compressedStr.length()+"\n"+compressedStr;
		
		printWriter.append(compressedStr);
		printWriter.flush();
	}
	
	public static String printMatrix(DoubleMatrix mat) {
		int rows = mat.rows;
		int cols = mat.columns;
		
		String matrix_str = rows+","+cols+",";
		
		double[][] mArray = mat.toArray2();
		for (int i = 0; i < rows - 1; i++) {
			String rowStr = Arrays.toString(mArray[i]);
			rowStr = rowStr.substring(1, rowStr.length() - 1) + ",";
			matrix_str += rowStr;
		}

		String lastRow = Arrays.toString(mArray[rows - 1]);
		
		lastRow = lastRow.substring(1, lastRow.length() - 1);
		matrix_str += lastRow;
		
		String compressedStr = compress(matrix_str);
		compressedStr = compressedStr.length()+"\n"+compressedStr;
		
		return compressedStr;
	}

	public static String printFormatMatrix(DoubleMatrix matrix) {
		int rows = matrix.rows;
		int cols = matrix.columns;
		String matrix_str = new String();

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				matrix_str = matrix_str + matrix.get(i, j) + "\t";
			}
			matrix_str += "\n";
		}

		return matrix_str;
	}

	/**
	 * compress a string
	 * 
	 * @param str
	 * @return
	 * @throws IOException
	 */
	public static String compress(String str) {
		try {
			
			ByteArrayOutputStream obj = new ByteArrayOutputStream();
			GZIPOutputStream gzip = new GZIPOutputStream(obj);

			gzip.write(str.getBytes("ISO-8859-1"));
			gzip.close();

			String outStr = obj.toString("ISO-8859-1");
			
			return outStr;
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}
	}

	/**
	 * decompress a string
	 * 
	 * @param str
	 * @return
	 */
	public static String decompress(String str) {
		try {

			GZIPInputStream gis = new GZIPInputStream(new ByteArrayInputStream(str.getBytes("ISO-8859-1")));
			BufferedReader bf = new BufferedReader(new InputStreamReader(gis, "ISO-8859-1"));

			String outStr = "";
			String line;

			while ((line = bf.readLine()) != null) {
				outStr += line;
			}

			return outStr;
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}

	}
	
	/**
	 * merge two matrices
	 * @param left
	 * @param right
	 * @return
	 */
	public static DoubleMatrix merge(DoubleMatrix left, DoubleMatrix right) {
		
		int row = left.rows;
		int column = left.columns + right.columns;
		
		DoubleMatrix matrix = new DoubleMatrix(row, column);
		for(int index=0; index<column; index++){
			if(index<left.columns)
				matrix.putColumn(index, left.getColumn(index));
			else
				matrix.putColumn(index, right.getColumn(index-left.columns));
		}
		
		return matrix;
	}

	/**
	 * transfer a matrix to a string.
	 * @param mat
	 * @return
	 */
	public static String matrixToString(DoubleMatrix mat){
		int rows = mat.rows;
		int cols = mat.columns;
		
		String matrix_str = rows+","+cols+",";
		
		double[][] mArray = mat.toArray2();
		for (int i = 0; i < rows - 1; i++) {
			String rowStr = Arrays.toString(mArray[i]);
			rowStr = rowStr.substring(1, rowStr.length() - 1) + ",";
			matrix_str += rowStr;
		}

		String lastRow = Arrays.toString(mArray[rows - 1]);
		
		lastRow = lastRow.substring(1, lastRow.length() - 1);
		matrix_str += lastRow;
		
		String compressedStr = compress(matrix_str);
		
		return compressedStr;
	}
	
	/**
	 * transfer a string to a matrix
	 * @param matrixString
	 * @return
	 */
	public static DoubleMatrix stringToMatrix(String matrixString){
		String matrix_str = decompress(matrixString);
		String[] matrixData = matrix_str.split(",");
		
		int m = Integer.parseInt(matrixData[0].trim());
		int n = Integer.parseInt(matrixData[1].trim());
		
		DoubleMatrix mat = new DoubleMatrix(m, n);
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {

				double m_ij = Double.parseDouble(matrixData[n * i + j+2].trim());
				mat.put(i, j, m_ij);
			}
		}

		return mat;
	}

	/**
	 * Write double[] to csv file
	 * @param output
	 * @param cols
	 * @param path
	 * @throws IOException
	 */
	public static void save2csv(double[] output, int cols, String path) throws IOException {
		System.out.println("Writing data to: " + path);
		StringBuilder builder = new StringBuilder();
		for (int i = 0; i < output.length / cols; i++)// for each row
		{
			for (int j = 0; j < cols; j++)// for each column
			{
				builder.append(output[i * cols + j] + "");// append to the output string
				if (j < cols - 1)// if this is not the last row element
					builder.append(",");// then add comma (if you don't like
										// commas you can use spaces)
			}
			builder.append("\n");// append new line at the end of the row
		}
		BufferedWriter writer = new BufferedWriter(new FileWriter(path));
		writer.write(builder.toString());// save the string representation of
											// the board
		writer.close();
	}
}
