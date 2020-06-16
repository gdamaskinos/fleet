/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package utils;

import org.jblas.DoubleMatrix;

import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.Random;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import Jama.Matrix;

/**
 * Created by Mercury on 2016/9/21.
 */

/**
 * Operations for Matrices on android (e.g. {@link Matrix}, {@link DoubleMatrix})
 */
public class MatrixOperation {

    /**
     * get a matrix from a stream
     * @param reader
     *              : the input stream
     * @return
     *              : matrix
     */
    public static Matrix readMatrix(BufferedReader reader){

        try {
            String m_read = reader.readLine();
            int m = Integer.parseInt(m_read);
            String n_read = reader.readLine();
            int n = Integer.parseInt(n_read);

            Matrix matrix = new Matrix(m, n);

            for(int i=0; i<m; i++){
                for(int j=0; j<n; j++){
                    String m_ij_read = reader.readLine();
                    double m_ij = Double.parseDouble(m_ij_read);

                    matrix.set(i, j, m_ij);
                }
            }

            //count += 1;

            return matrix;

        } catch (IOException e){
            e.printStackTrace();
            return null;
        }
    }

    public static Matrix readMatrix(int length, BufferedReader reader){
        try{
            int i, j;

            int dataLen = length;
            char[] data = new char[dataLen];

            reader.read(data, 0, dataLen);
            String compressedStr = new String(data);

            String matrix_str = decompress(compressedStr);

            String[] matrixData = matrix_str.split(",");

            int m = Integer.parseInt(matrixData[0].trim());
            int n = Integer.parseInt(matrixData[1].trim());

            Matrix mat = new Matrix(m, n);
            for(i=0; i<m; i++){
                for(j=0; j<n; j++){

                    double m_ij = Double.parseDouble(matrixData[n * i + j+2].trim());
                    mat.set(i, j, m_ij);

                }
            }

            return mat;

        } catch (IOException e){
            e.printStackTrace();
            return null;
        }
    }

    /**
     * print a matrix
     * @param matrix
     *              : the matrix needed to be printed
     * @return
     *              : the format string representing the matrix
     */
    public static String printFormatMatrix(Matrix matrix){
        int rows = matrix.getRowDimension();
        int cols = matrix.getColumnDimension();
        String matrix_str = new String();

        for(int i=0; i<rows; i++){
            for(int j=0; j<cols; j++){
                matrix_str = matrix_str + matrix.get(i, j)+"\t";
            }
            matrix_str += "\n";
        }

        return matrix_str;
    }

    /**
     * print a matrix
     * @param matrix
     *              : the matrix needed to be printed
     * @return
     *              : the string representing the matrix
     */
    /*public static String printMatrix(Matrix matrix){

        double startPrintTime = System.currentTimeMillis();

        int rows = matrix.getRowDimension();
        int cols = matrix.getColumnDimension();
        String matrix_str = new String();

        matrix_str = rows + "\n" + cols + "\n";

        for(int i=0; i<rows; i++){
            for(int j=0; j<cols; j++) {
                matrix_str = matrix_str + matrix.get(i, j) + "\n";
            }
        }

        double printTime = System.currentTimeMillis()-startPrintTime;

        System.out.println("Print matrix time cost "+printTime+" ms");

        return matrix_str;
    }*/

    public static String printMatrix(Matrix matrix){

        int rows = matrix.getRowDimension();
        int cols = matrix.getColumnDimension();
        String matrix_str = new String();

        matrix_str = rows+","+cols+",";


        double[][] mArray = matrix.getArray();
        for(int i=0; i<rows-1; i++){
            String rowStr = Arrays.toString(mArray[i]);
            rowStr = rowStr.substring(1, rowStr.length()-1)+",";
            matrix_str += rowStr;
        }

        String lastRow = Arrays.toString(mArray[rows-1]);

        lastRow = lastRow.substring(1, lastRow.length()-1);

        matrix_str += lastRow;

        //String compressedStr = compress(matrix_str);
        //compressedStr = compressedStr.length()+"\n"+compressedStr;

        return matrix_str.length()+"\n"+matrix_str;
    }

    /**
     * Get subarray. Pay attention to index ranges following MATLAB's notation.
     *
     * @param matrix
     *            : DoubleMatrix ( M x N )
     * @param row
     *            : 0 <= row < M (normal row pointer)
     * @param start
     *            : 1 <= start < N (matlab style)
     * @param end
     *            : start < end < N (matlab style)
     * @param
     *            : step
     * @return
     *            : A single row-matrix.
     */
    public static Matrix getRowRange(Matrix matrix, int row, int start, int step, int end){
        int size = 0;
        for (int i = start - 1; i < end; i += step)
            size++;

        Matrix res = new Matrix(1, size);
        int j=0;
        for (int i = start - 1; i < end; i += step, j++) {
            res.set(0, j, matrix.get(row, i));
        }

        return res;
    }

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
        // System.out.println("Size: "+ size);
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
    public static Matrix gFunction(Matrix X, Matrix Y){
        int rows = X.getRowDimension();
        int cols = X.getColumnDimension();

        Matrix res = new Matrix(rows, cols);

        for(int i=0; i<rows; i++){
            for(int j=0; j<cols; j++){
                res.set(i, j, X.get(i, j) / (1 + Math.exp(-Y.get(i, j))));
            }
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
        DoubleMatrix res = new DoubleMatrix(X.rows, X.columns);
        for (int i = 0; i < X.length; i++)
            res.put(i, X.get(i) / (1 + Math.exp(-Y.get(i))));
        return res;
    }

    /**
     * Computes distance between two column vectors
     * @param x
     *          : a real vector (N X 1)
     * @param y
     *          : a real vector (N X 1)
     * @return
     *          : distance between x and y
     */
    public static double distance(Matrix x, Matrix y){
        int length = x.getRowDimension();
        double distance = 0;

        for(int i=0; i<length; i++){
            distance += Math.pow(x.get(i, 0)-y.get(i, 0), 2.0);
        }

        return distance;
    }

    public static Matrix getRowAt(Matrix matrix, int index){
        int length = matrix.getColumnDimension();
        Matrix row = new Matrix(1, length);
        for(int i=0; i<length; i++)
            row.set(0, i, matrix.get(index, i));

        return row;
    }

    public static void setRowAt(Matrix matrix, int index, Matrix row){
        int length = row.getColumnDimension();
        for(int i=0; i<length; i++)
            matrix.set(index, i, row.get(0, i));
    }

    /**
     * initial the matrix so that its entries' values follow
     * standard normal distribution
     * @param matrix
     * @return matrix
     */
    public static Matrix randn(Matrix matrix){
        int rows = matrix.getRowDimension();
        int cols = matrix.getColumnDimension();

        Random random = new Random();

        for(int i=0; i<rows; i++){
            for(int j=0; j<cols; j++){
                matrix.set(i, j, random.nextGaussian());
            }
        }

        return matrix;
    }

    /**
     * Generate a new matrix which has the given number of replications of this.
     * @param matrix
     * @param rowMult
     * @param colMult
     * @return matrixMult
     */
    public static Matrix repmat(Matrix matrix, int rowMult, int colMult){
        int rows = matrix.getRowDimension();
        int cols = matrix.getColumnDimension();

        Matrix matrixMult = new Matrix(rows*rowMult, cols*colMult);

        for(int rowth=0; rowth<rowMult; rowth++){
            for(int colth=0; colth<colMult; colth++){
                for(int i=0; i<rows; i++){
                    for(int j=0; j<cols; j++){
                        matrixMult.set(rowth*rows+i, colth*cols+j, matrix.get(i, j));
                    }
                }
            }
        }

        return matrixMult;
    }

    /**
     * check whether there is NaN value in the matrix
     * @param matrix
     * @return
     *         : no NaN value => valid=false; otherwise valid=true
     */
    public static boolean isNaNMatrix(Matrix matrix){
        int rows = matrix.getRowDimension();
        int cols = matrix.getColumnDimension();

        for(int i=0; i<rows; i++){
            for(int j=0; j<cols; j++){
                if(Double.isNaN(matrix.get(i, j)))
                    return true;
            }
        }

        return false;
    }

    /**
     * compress a string
     * @param str
     * @return
     * @throws IOException
     */
    public static String compress(String str){
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
     * @param str
     * @return
     */
    public static String decompress(String str){
        try {

            GZIPInputStream gis = new GZIPInputStream(new ByteArrayInputStream(str.getBytes("ISO-8859-1")));
            BufferedReader bf = new BufferedReader(new InputStreamReader(gis, "ISO-8859-1"));

            String outStr = bf.readLine();
            //String line;

            //while ((line = bf.readLine()) != null) {
                //outStr += line;
            //}

            return outStr;
        } catch (IOException e){

            System.out.println("Error happens");
            e.printStackTrace();
            return null;
        }


    }
}
