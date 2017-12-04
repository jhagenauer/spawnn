package gwr.ga;

import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryFactory;

import spawnn.utils.DataUtils;
import spawnn.utils.GeoUtils.GWKernel;

public class GWR_Test {

	private static Logger log = Logger.getLogger(GWR_Test.class);
	int threads = Math.max(1, Runtime.getRuntime().availableProcessors() - 1);;

	public static void main(String[] args) {
		GeometryFactory gf = new GeometryFactory();
		GWKernel kernel = GWKernel.boxcar;
		int[] ga = new int[] { 0, 1 };
		int[] fa = new int[] { 3 };
		int ta = 4;

		int pointsPerRow = 24;
		List<double[]> samples = new ArrayList<double[]>(BuildTestData.createSpDepTestData(pointsPerRow));
		List<Geometry> geoms = new ArrayList<Geometry>();
		for (double[] d : samples)
			geoms.add(gf.createPoint(new Coordinate(d[ga[0]], d[ga[1]])));
		log.debug(samples.size());
		DataUtils.writeCSV("output/spDat.csv", samples, new String[] { "long", "lat", "beta", "x1", "y" });

		List<Double> bw = new ArrayList<Double>();
		for (int i = 0; i < samples.size(); i++)
			bw.add(0.1091758);
		GWRIndividual ind = new GWRIndividual(bw, 0);

		GWRCostCalculator ccAICc = new GWRIndividualCostCalculator_AICc(samples, fa, ga, ta, kernel,false);

		double aicc = ccAICc.getCost(ind);
		log.debug(aicc);
	}
}
