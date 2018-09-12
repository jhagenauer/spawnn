package gwr.ga;

import java.util.List;
import java.util.Map;

import heuristics.ga.GAIndividual;
import spawnn.dist.Dist;

public interface GWRIndividual<T> extends GAIndividual<T> {
	public Map<double[],Double> getSpatialBandwidth(List<double[]> samples, Dist<double[]> gDist);
	public String geneToString(int i);
}
