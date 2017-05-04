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

import javax.imageio.ImageIO;

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

public class LandConsumption_cv2 {

	private static Logger log = Logger.getLogger(LandConsumption_cv2.class);

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

		List<double[]> samplesTrain = new ArrayList<>();
		List<double[]> desiredTrain = new ArrayList<>();
		List<double[]> samplesVal = new ArrayList<>();
		List<double[]> desiredVal = new ArrayList<>();

		for (double[] d : samples) {
			if (r.nextDouble() < 0.8) {
				samplesTrain.add(d);
				desiredTrain.add(new double[] { d[ta] });
			} else {
				samplesVal.add(d);
				desiredVal.add(new double[] { d[ta] });
			}
		}

		boolean gaussian = true;
		boolean adaptive = true;

		int t_max = 1000000;

		int aMax = 100;
		int lambda = 1000;
		double alpha = 0.5;
		double beta = 0.000005;

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
			 
			/*if( t % 10000 == 0 ) {
				List<double[]> responseVal = new ArrayList<double[]>();
				for (int i = 0; i < samplesVal.size(); i++)
					responseVal.add(llm.present(samplesVal.get(i)));
				log.debug(t+" " + SupervisedUtils.getRMSE(responseVal, desiredVal));
				
				
				Map<double[],Set<double[]>> mTrain = NGUtils.getBmuMapping(samplesTrain, llm.neurons, sorter);
				Map<double[],Set<double[]>> mVal = NGUtils.getBmuMapping(samplesVal, llm.neurons, sorter);
				log.debug(t+" " + DataUtils.getMeanQuantizationError(mTrain, gDist) + "\t" + DataUtils.getMeanQuantizationError(mVal, gDist));
				log.debug(t+" "+llm.neurons.size());
			}*/
		}

		List<double[]> responseVal = new ArrayList<double[]>();
		for (int i = 0; i < samplesVal.size(); i++)
			responseVal.add(llm.present(samplesVal.get(i)));

		log.debug("incLLM RMSE: " + SupervisedUtils.getRMSE(responseVal, desiredVal));
				
		{
			Map<double[],Set<double[]>> mTrain = NGUtils.getBmuMapping(samplesTrain, llm.neurons, sorter);
			Map<double[],Set<double[]>> mVal = NGUtils.getBmuMapping(samplesVal, llm.neurons, sorter);
					
			List<Set<double[]>> lTrain = new ArrayList<>();
			List<Set<double[]>> lVal = new ArrayList<>();
			for( double[] d : mTrain.keySet() ) { 
				lTrain.add(mTrain.get(d));
				lVal.add(mVal.get(d));
			}			
			LinearModel lm = new LinearModel( samplesTrain, lTrain, fa, ta, false);
			List<Double> pred = lm.getPredictions(samplesVal, lVal, fa);
			log.debug("incLLM-LM RMSE: "+SupervisedUtils.getRMSE(pred, samplesVal, ta));
		}
		
		
		log.debug("neurons: "+llm.neurons.size());
		
		geoDrawWithOverlay(sdf.geoms, llm.neurons, ga, "data/incllm.png");
		
		/*for( double[] n : llm.neurons ) {
			List<double[]> nbs = new ArrayList<>(Connection.getNeighbors(llm.cons.keySet(), n, 1));
			log.debug("n: "+n.hashCode());
			for( int i = 0; i < nbs.size(); i++ )
				log.debug(i+" "+nbs.get(i).hashCode() );
		}*/
				
		for (double bw : new double[]{ 8 }) {

			Map<double[], Double> bandwidth = new HashMap<>();
			for (double[] a : samples) {
				if (!adaptive)
					bandwidth.put(a, bw);
				else {
					int k = (int) bw;
					List<double[]> s = new ArrayList<>(samples);
					Collections.sort(s, new Comparator<double[]>() {
						@Override
						public int compare(double[] o1, double[] o2) {
							return Double.compare(gDist.dist(o1, a), gDist.dist(o2, a));
						}
					});
					bandwidth.put(a, gDist.dist(s.get(k - 1), a));
				}
			}

			DoubleMatrix Y = new DoubleMatrix(LinearModel.getY(samplesTrain, ta));
			DoubleMatrix X = new DoubleMatrix(LinearModel.getX(samplesTrain, fa, true));

			DoubleMatrix XVal = new DoubleMatrix(LinearModel.getX(samplesVal, fa, true));
			List<Double> predictions = new ArrayList<>();
			for (int i = 0; i < samplesVal.size(); i++) {
				double[] a = samplesVal.get(i);

				DoubleMatrix XtW = new DoubleMatrix(X.getColumns(), X.getRows());
				for (int j = 0; j < X.getRows(); j++) {
					double[] b = samplesTrain.get(j);
					double d = gDist.dist(a, b);

					double w;
					if (gaussian) // Gaussian
						w = Math.exp(-0.5 * Math.pow(d / bandwidth.get(a), 2));
					else // bisquare
						w = Math.pow(1.0 - Math.pow(d / bandwidth.get(a), 2), 2);
					XtW.putColumn(j, X.getRow(j).mul(w));
				}
				DoubleMatrix XtWX = XtW.mmul(X);
				DoubleMatrix beta2 = Solve.solve(XtWX, XtW.mmul(Y));

				predictions.add(XVal.getRow(i).mmul(beta2).get(0));
			}
			log.debug("GWR: "+bw + "," + SupervisedUtils.getRMSE(predictions, samplesVal, ta));
		}
		
		// lm --------------
		
		{
		LinearModel lm = new LinearModel(samplesTrain, fa, ta, false);
		List<Double> pred = lm.getPredictions(samplesVal, fa);
		log.debug("LM: "+SupervisedUtils.getRMSE(pred, samplesVal, ta));
		}
		
		// k-means ------------
		
		{
			Map<double[], Set<double[]>> mTrain = Clustering.kMeans(samplesTrain, llm.neurons.size(), gDist, 0.000001);
			Map<double[],Set<double[]>> mVal = new HashMap<>();
			for( double[] d : samplesVal ) {
				double[] bestK = null;
				for( double[] k : mTrain.keySet() )
					if( bestK == null || gDist.dist(d, k) < gDist.dist( d, bestK ) )
						bestK = k;
				if( !mVal.containsKey( bestK ))
					mVal.put(bestK, new HashSet<double[]>() );
				mVal.get(bestK).add(d);
			}
			log.debug("kMeans: " + DataUtils.getMeanQuantizationError(mTrain, gDist) + "\t" + DataUtils.getMeanQuantizationError(mVal, gDist));
					
			List<Set<double[]>> lTrain = new ArrayList<>();
			List<Set<double[]>> lVal = new ArrayList<>();
			for( double[] d : mTrain.keySet() ) { 
				lTrain.add(mTrain.get(d));
				lVal.add(mVal.get(d));
			}			
			LinearModel lm = new LinearModel( samplesTrain, lTrain, fa, ta, false);
			List<Double> pred = lm.getPredictions(samplesVal, lVal, fa);
			log.debug("kMeans LM: "+SupervisedUtils.getRMSE(pred, samplesVal, ta));
		}
	}
	
	public static void geoDrawWithOverlay( List<Geometry> geoms, List<double[]> fg, int[] ga, String fn ) {
		try {			
			StyleBuilder sb = new StyleBuilder();
			MapContent map = new MapContent();
			ReferencedEnvelope maxBounds = null;
			
			{ // background
				SimpleFeatureTypeBuilder typeBuilder = new SimpleFeatureTypeBuilder();
				typeBuilder.setName("bg");
				typeBuilder.setCRS(null);
	
				if (geoms.get(0) instanceof Polygon)
					typeBuilder.add("the_geom", Polygon.class);
				else if (geoms.get(0) instanceof Point)
					typeBuilder.add("the_geom", Point.class);
				else if (geoms.get(0) instanceof MultiPolygon)
					typeBuilder.add("the_geom", MultiPolygon.class);
				else
					throw new RuntimeException("Unknown Geometry type: " + geoms.get(0));
				SimpleFeatureBuilder featureBuilder = new SimpleFeatureBuilder(typeBuilder.buildFeatureType());
				
				DefaultFeatureCollection fc = new DefaultFeatureCollection();
				for (int i = 0; i < geoms.size(); i++ ) {
					featureBuilder.add(geoms.get(i));
					fc.add(featureBuilder.buildFeature("" + fc.size()));
				}			
				maxBounds = fc.getBounds();
	
				Color color = Color.GRAY;
				if (geoms.get(0) instanceof Polygon || geoms.get(0) instanceof MultiPolygon)
					map.addLayer(new FeatureLayer(fc, SLD.wrapSymbolizers(sb.createPolygonSymbolizer(color, Color.BLACK, 1.0))));
				else if (geoms.get(0) instanceof Point) {
					Mark mark = sb.createMark(StyleBuilder.MARK_CIRCLE, color);
					map.addLayer(new FeatureLayer(fc, SLD.wrapSymbolizers(sb.createPointSymbolizer(sb.createGraphic(null, mark, null)))));
				} else
					throw new RuntimeException("No layer for geometry type added");
			}
			
			{ // foreground
				SimpleFeatureTypeBuilder typeBuilder = new SimpleFeatureTypeBuilder();
				typeBuilder.setName("bg");
				typeBuilder.setCRS(null);
				typeBuilder.add("the_geom", Point.class);
				SimpleFeatureBuilder featureBuilder = new SimpleFeatureBuilder(typeBuilder.buildFeatureType());
				
				DefaultFeatureCollection fc = new DefaultFeatureCollection();
				GeometryFactory gf = new GeometryFactory();
				for (double[] d : fg ) {
					featureBuilder.add(gf.createPoint( new Coordinate(d[ga[0]], d[ga[1]])) );
					fc.add(featureBuilder.buildFeature("" + fc.size()));
				}			
				maxBounds.expandToInclude(fc.getBounds());
	
				Color color = Color.RED;
				Mark mark = sb.createMark(StyleBuilder.MARK_CIRCLE, color);
				map.addLayer(new FeatureLayer(fc, SLD.wrapSymbolizers(sb.createPointSymbolizer(sb.createGraphic(null, mark, null)))));
			}

			GTRenderer renderer = new StreamingRenderer();
			RenderingHints hints = new RenderingHints(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
			hints.put(RenderingHints.KEY_ALPHA_INTERPOLATION, RenderingHints.VALUE_ALPHA_INTERPOLATION_QUALITY);
			hints.put(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_SPEED);
			renderer.setMapContent(map);

			Rectangle imageBounds = null;
			try {
				double heightToWidth = maxBounds.getSpan(1) / maxBounds.getSpan(0);
				int imageWidth = 2000;

				imageBounds = new Rectangle(0, 0, imageWidth, (int) Math.round(imageWidth * heightToWidth));
				// imageBounds = new Rectangle( 0, 0, mp.getWidth(), (int) Math.round(mp.getWidth() * heightToWidth));

				BufferedImage image = new BufferedImage(imageBounds.width, imageBounds.height, BufferedImage.TYPE_INT_RGB);

				// png
				Graphics2D gr = image.createGraphics();
				gr.setPaint(Color.WHITE);
				gr.fill(imageBounds);

				renderer.paint(gr, imageBounds, maxBounds);

				ImageIO.write(image, "png", new FileOutputStream(fn));
				image.flush();
			} catch (IOException e) {
				e.printStackTrace();
			}
			map.dispose();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
