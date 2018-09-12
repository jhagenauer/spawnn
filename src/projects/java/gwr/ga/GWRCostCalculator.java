package gwr.ga;

import java.util.List;

import heuristics.CostCalculator;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.utils.GeoUtils.GWKernel;

public abstract class GWRCostCalculator<T extends GWRIndividual<T>> implements CostCalculator<T> {

	protected List<double[]> samples;
	protected int[] fa;
	protected int ta;
	protected GWKernel kernel;
	protected Dist<double[]> gDist;
	protected boolean adaptive;

	GWRCostCalculator(List<double[]> samples, int[] fa, int[] ga, int ta, GWKernel kernel) {
		this.samples = samples;
		this.fa = fa;
		this.gDist = new EuclideanDist(ga);
		this.kernel = kernel;
		this.ta = ta;
	}
}