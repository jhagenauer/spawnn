package inc_llm;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.Map.Entry;

import javax.imageio.ImageIO;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.log4j.Logger;
import org.geotools.feature.DefaultFeatureCollection;
import org.geotools.feature.simple.SimpleFeatureBuilder;
import org.geotools.feature.simple.SimpleFeatureTypeBuilder;
import org.geotools.geometry.jts.ReferencedEnvelope;
import org.geotools.map.FeatureLayer;
import org.geotools.map.MapContent;
import org.geotools.renderer.GTRenderer;
import org.geotools.renderer.lite.StreamingRenderer;
import org.geotools.styling.Mark;
import org.geotools.styling.SLD;
import org.geotools.styling.StyleBuilder;
import org.jblas.DoubleMatrix;
import org.jblas.Solve;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryFactory;
import com.vividsolutions.jts.geom.MultiPolygon;
import com.vividsolutions.jts.geom.Point;
import com.vividsolutions.jts.geom.Polygon;

import chowClustering.LinearModel;
import nnet.SupervisedUtils;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.utils.NGUtils;
import spawnn.som.decay.ConstantDecay;
import spawnn.som.decay.LinearDecay;
import spawnn.som.decay.PowerDecay;
import spawnn.utils.Clustering;
import spawnn.utils.DataUtils;
import spawnn.utils.DataUtils.Transform;
import spawnn.utils.SpatialDataFrame;

public class LandConsumption_cv3 {

	private static Logger log = Logger.getLogger(LandConsumption_cv3.class);

	public static void main(String[] args) {
		Random r = new Random(0);
		
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/election/election2004.shp"), true);
		for (int i = 0; i < sdf.samples.size(); i++) {
			Point p = sdf.geoms.get(i).getCentroid();
			sdf.samples.get(i)[0] = p.getX();
			sdf.samples.get(i)[1] = p.getY();
		}

		int[] ga = new int[] { 0, 1 };
		int[] fa = new int[] { 52, 49, 10 };
		int ta = 7;
		Dist<double[]> gDist = new EuclideanDist(ga);

		List<double[]> samples = sdf.samples;
		DataUtils.transform(samples, fa, Transform.zScore);
		
		int t_max = 1000000;

		int aMax = 100;
		int lambda = 1000;
		double alpha = 0.5;
		double beta = 0.000005;
		
		List<Entry<List<Integer>, List<Integer>>> cvList = SupervisedUtils.getCVList(10, 1, samples.size());
		SummaryStatistics ss = new SummaryStatistics();
		for (final Entry<List<Integer>, List<Integer>> cvEntry : cvList) {
			List<double[]> samplesTrain = new ArrayList<double[]>();
			List<double[]> desiredTrain = new ArrayList<double[]>();
			for( int k : cvEntry.getKey() ) {
				samplesTrain.add(samples.get(k));
				desiredTrain.add(new double[]{samples.get(k)[ta]});
			}
			
			List<double[]> samplesVal = new ArrayList<double[]>();
			List<double[]> desiredVal = new ArrayList<double[]>();
			for( int k : cvEntry.getValue() ) {
				samplesVal.add(samples.get(k));
				desiredVal.add(new double[]{samples.get(k)[ta]});
			}
			
			List<double[]> neurons = new ArrayList<double[]>();
			for (int i = 0; i < 2; i++) {
				double[] d = samples.get(r.nextInt(samples.size()));
				neurons.add(Arrays.copyOf(d, d.length));
			}

			Sorter<double[]> sorter = new DefaultSorter<double[]>(gDist);
			
			IncLLM llm = new IncLLM(neurons, 
					new ConstantDecay(0.01), 
					new ConstantDecay(0.001), 
					new PowerDecay(0.1, 5.0E-5),  
					new PowerDecay(0.01, 1.0E-5), 
					sorter, aMax, lambda, alpha, beta, fa, 1, t_max);
			for (int t = 0; t < t_max; t++) {
				int idx = r.nextInt(samplesTrain.size());
				llm.train(t, samplesTrain.get(idx), desiredTrain.get(idx));
			}

			List<double[]> responseVal = new ArrayList<double[]>();
			for (int i = 0; i < samplesVal.size(); i++)
				responseVal.add(llm.present(samplesVal.get(i)));

			ss.addValue( SupervisedUtils.getRMSE(responseVal, desiredVal));
		}
		log.debug(ss.getMean());
	}
}
