/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package utils.dl4j;

import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Fixes flattening order bug {@link DefaultGradient}
 *
 */
public class MyDefaultGradient implements Gradient {

	private static final long serialVersionUID = 1L;
	private static final char DEFAULT_FLATTENING_ORDER = 'f';
    private Map<String,INDArray> gradients = new LinkedHashMap<>();
    private Map<String,Character> flatteningOrders;
    private INDArray flattenedGradient;

    public MyDefaultGradient(){ }

    public MyDefaultGradient(INDArray flattenedGradient){
        this.flattenedGradient = flattenedGradient;
    }

    @Override
    public Map<String, INDArray> gradientForVariable() {
        return gradients;
    }

    @Override
    public INDArray gradient(List<String> order) {
        List<INDArray> toFlatten = new ArrayList<>();
        if(flatteningOrders == null) {
            for (String s : order) {
                if (!gradients.containsKey(s)) continue;
                toFlatten.add(gradients.get(s));
            }
        } else {
            for(String s : order){
                if (!gradients.containsKey(s)) continue;
                if (flatteningOrders.containsKey(s) && flatteningOrders.get(s) != DEFAULT_FLATTENING_ORDER) {
                    //Arrays with non-default order get flattened to row vector first, then everything is flattened to f order
                    //TODO revisit this, and make more efficient
                    toFlatten.add(Nd4j.toFlattened(flatteningOrders.get(s),gradients.get(s)));
                } else {
                    toFlatten.add(gradients.get(s));
                }
            }
        }
        return Nd4j.toFlattened(DEFAULT_FLATTENING_ORDER, toFlatten);
    }

    private void flattenGradient(){
        if(flatteningOrders != null){
            //Arrays with non-default order get flattened to row vector first, then everything is flattened to f order
            //TODO revisit this, and make more efficient
            List<INDArray> toFlatten = new ArrayList<>();
            for(Map.Entry<String,INDArray> entry : gradients.entrySet()){
                if(flatteningOrders.containsKey(entry.getKey()) && flatteningOrders.get(entry.getKey()) != DEFAULT_FLATTENING_ORDER){
                    //Specific flattening order for this array, that isn't the default
                    toFlatten.add(Nd4j.toFlattened(flatteningOrders.get(entry.getKey()),entry.getValue()));
                } else {
                    //default flattening order for this array
                    toFlatten.add(entry.getValue());
                }
            }
            flattenedGradient = Nd4j.toFlattened(DEFAULT_FLATTENING_ORDER, toFlatten);
        } else {
            //Standard case: flatten all to f order
            flattenedGradient = Nd4j.toFlattened(DEFAULT_FLATTENING_ORDER, gradients.values());
        }
    }

    @Override
    public INDArray gradient() {
        flattenGradient(); // fixed bug
        if(flattenedGradient != null) return flattenedGradient;
        return flattenedGradient;
    }

    @Override
    public void clear() {
        gradients.clear();
    }

    @Override
    public INDArray getGradientFor(String variable) {
        return gradients.get(variable);
    }

    @Override
    public INDArray setGradientFor(String variable, INDArray newGradient) {
        INDArray last = gradients.put(variable, newGradient);
        // TODO revisit whether setGradientFor should update the gradient that can be pulled from this object in any form - currently does not update flattened
        // use of unitialized var for flattengradient in backprop is generating an error in gradient calc if bellow is used
//        flattenGradient();
        return last;
    }

    @Override
    public INDArray setGradientFor(String variable, INDArray gradient, Character flatteningOrder) {
        INDArray last = setGradientFor(variable,gradient);

        if(flatteningOrder != null){
            if(flatteningOrders == null) flatteningOrders = new LinkedHashMap<>();
            flatteningOrders.put(variable,flatteningOrder);
        }
        return last;
    }

    @Override
    public Character flatteningOrderForVariable(String variable) {
        if(flatteningOrders == null) {
        	System.out.println("IS NULL");
        	return null;
        }
        
        return flatteningOrders.get(variable);
    }


    @Override
    public String toString() {
        return "DefaultGradient{" +
                "gradients=" + gradients +
                (flatteningOrders != null ? flatteningOrders : "") +
                '}';
    }
}
