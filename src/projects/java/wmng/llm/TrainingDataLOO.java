package wmng.llm;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

class TrainingDataLOO {
	List<double[]> samplesTrain;
	List<double[]> desiredTrain;
	Map<double[],Map<double[],Double>> dMapTrain;
	
	double[] sampleVal, desiredVal;
	Map<double[],Map<double[],Double>> dMapVal;
	
	public TrainingDataLOO( int idx, List<double[]> samples, List<double[]>  desired, Map<double[],Map<double[],Double>> dMap ) {
		samplesTrain = new ArrayList<double[]>(samples);
		desiredTrain = new ArrayList<double[]>(desired);
						
		sampleVal = samplesTrain.remove(idx);
		desiredVal = desiredTrain.remove(idx);
		
		dMapVal = dMap;		
		dMapTrain = getDMapWithout(dMap, sampleVal);
	}
	
	private Map<double[],Map<double[],Double>> getDMapWithout( Map<double[],Map<double[],Double>> dMap, double[] d ) {
		Map<double[],Map<double[],Double>> r = new HashMap<double[],Map<double[],Double>>();
		for( double[] a : dMap.keySet() ) {
			if( a == d )
				continue;
			Map<double[],Double> nm = new HashMap<double[],Double>(dMap.get(a));
			nm.remove(d);
			r.put(a, nm);
		}
		return r;
	}
}
