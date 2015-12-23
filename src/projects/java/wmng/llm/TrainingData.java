package wmng.llm;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class TrainingData {
	List<double[]> samplesTrain = new ArrayList<double[]>();
	List<double[]> desiredTrain = new ArrayList<double[]>();
	Map<double[],Map<double[],Double>> dMapTrain;
	
	List<double[]> samplesVal = new ArrayList<double[]>();
	List<double[]> desiredVal = new ArrayList<double[]>();
	Map<double[],Map<double[],Double>> dMapVal;
	
	public TrainingData(List<double[]> samples, List<double[]>  desired, Map<double[],Map<double[],Double>> dMap ) {
		samplesTrain = samples;
		desiredTrain = desired;
		dMapTrain = dMap;
		
		samplesVal = samples;
		desiredVal = desired;
		dMapVal = dMap;
	}
}