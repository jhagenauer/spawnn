package llm_cng;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import javax.imageio.ImageIO;

import llm.ErrorBmuGetter;
import llm.ErrorSorter;
import llm.LLMNG;
import llm.LLMSOM;

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
import org.opengis.referencing.crs.CoordinateReferenceSystem;

import rbf.Meuse;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.utils.NGUtils;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.bmu.DefaultBmuGetter;
import spawnn.som.bmu.KangasBmuGetter;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.GridPos;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.utils.SomUtils;
import spawnn.utils.ColorBrewerUtil;
import spawnn.utils.ColorBrewerUtil.ColorMode;
import spawnn.utils.DataUtils;
import spawnn.utils.Drawer;
import spawnn.utils.SpatialDataFrame;

import com.vividsolutions.jts.geom.Geometry;

public class HousepriceTest {

	private static Logger log = Logger.getLogger(HousepriceTest.class);

	public static void main(String[] args) {
		final DecimalFormat df = new DecimalFormat("00");
		final Random r = new Random();
		final int T_MAX = 100000;

		final List<double[]> samples = new ArrayList<double[]>();
		final List<Geometry> geoms = new ArrayList<Geometry>();
		final List<double[]> desired = new ArrayList<double[]>();

		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromCSV(new File("data/marco/dat4/gwr.csv"), new int[] { 6, 7 }, new int[] {}, true);
		List<String> vars = new ArrayList<String>();
		vars.add("xco");
		vars.add("yco");
		vars.add("lnarea_tot");
		vars.add("lnarea_plo");
		vars.add("attic_dum");
		vars.add("cellar_dum");
		vars.add("cond_house_3");
		vars.add("heat_3");
		vars.add("bath_3");
		vars.add("garage_3");
		vars.add("terr_dum");
		vars.add("age_num");
		vars.add("time_index");
		vars.add("zsp_alq_09");
		vars.add("gem_kauf_i");
		vars.add("gem_abi");
		vars.add("gem_alter_");
		vars.add("ln_gem_dic");

		final int da = sdf.names.indexOf("lnp");
		for (double[] d : sdf.samples) {
			if (d[sdf.names.indexOf("time_index")] < 6)
				continue;
			int idx = sdf.samples.indexOf(d);
			double[] nd = new double[vars.size()];
			
			for (int i = 0; i < nd.length; i++)
				nd[i] = d[sdf.names.indexOf(vars.get(i))]; 
					
			samples.add(nd);
			desired.add(new double[]{d[da]});
			geoms.add(sdf.geoms.get(idx));
		}

		final int[] fa = new int[vars.size()-2]; // omit geo-vars
		for( int i = 0; i < fa.length; i++ )
			fa[i] = i+2;
		final int[] ga = new int[] { 0, 1 };

		// ------------------------------------------------------------------------

		{ // just write it to csv inkl. csv for external use
			List<double[]> l = new ArrayList<double[]>();
			for( double[] d : samples ) {
				double[] nd = Arrays.copyOf(d, d.length+1);
				nd[nd.length-1] = desired.get(samples.indexOf(d))[0];
				l.add(nd);
			}
			List<String> nv = new ArrayList<String>(vars);
			nv.add("lnp");
			DataUtils.writeCSV("output/houseprice.csv", l, nv.toArray(new String[] {}));
		}
		DataUtils.zScoreColumns(samples, fa);
		DataUtils.zScoreColumn(samples, da); // should not be necessary
	
		final Dist<double[]> gDist = new EuclideanDist(ga);
		final Dist<double[]> fDist = new EuclideanDist(fa);

		for (int run = 0; run < 1; run++)
			for (int l = 3; l <= 3; l++) {

				List<double[]> neurons = new ArrayList<double[]>();
				for (int i = 0; i < 60; i++) {
					double[] d = samples.get(r.nextInt(samples.size()));
					neurons.add(Arrays.copyOf(d, d.length));
				}

				ErrorSorter errorSorter = new ErrorSorter(samples, desired);
				DefaultSorter<double[]> gSorter = new DefaultSorter<>(gDist);
				Sorter<double[]> sorter = new KangasSorter<>(gSorter, errorSorter, l);
				LLMNG ng = new LLMNG(neurons, 
						neurons.size()/3, 1, 0.5, 0.005, 
						neurons.size()/3, 1, 0.1, 0.005, 
						sorter, fa, 1);
				errorSorter.setLLMNG(ng);

				for (int t = 0; t < T_MAX; t++) {
					int j = r.nextInt(samples.size());
					ng.train((double) t / T_MAX, samples.get(j), desired.get(j));
				}
				Map<double[], Set<double[]>> mapping = NGUtils.getBmuMapping(samples, ng.getNeurons(), sorter);

				List<double[]> response = new ArrayList<double[]>();
				for (double[] x : samples)
					response.add(ng.present(x));
				log.debug(l+", RMSE: " + Meuse.getRMSE(response, desired) + ", R2: " + Math.pow(Meuse.getPearson(response, desired), 2));
				
				Map<double[],double[]> geoCoefs = new HashMap<double[],double[]>();
				for( double[] n : ng.getNeurons() ) {
					double[] cg = new double[2+fa.length];
					cg[0] = n[ga[0]];
					cg[1] = n[ga[1]];
					for( int i = 0; i < fa.length; i++ )
						cg[2+i] = ng.matrix.get(n)[0][i];
					geoCoefs.put(n, cg);
				}
				DataUtils.writeCSV("output/coefs_"+df.format(l)+"_"+df.format(run)+".csv", geoCoefs.values(), vars.toArray( new String[]{} ));
	
			}
	}

	public static void geoDrawValues(List<Geometry> geoms, List<Double> values, CoordinateReferenceSystem crs, ColorMode cm, String fn) {
		SimpleFeatureTypeBuilder typeBuilder = new SimpleFeatureTypeBuilder();
		typeBuilder.setName("data");
		typeBuilder.setCRS(crs);
		typeBuilder.add("the_geom", geoms.get(0).getClass());

		SimpleFeatureBuilder featureBuilder = new SimpleFeatureBuilder(typeBuilder.buildFeatureType());
		Map<Geometry, Double> m = new HashMap<Geometry, Double>();
		for (int i = 0; i < geoms.size(); i++)
			m.put(geoms.get(i), values.get(i));

		Map<Geometry, Color> colMap = ColorBrewerUtil.valuesToColors(m, cm);
		Set<Color> cols = new HashSet<Color>(colMap.values());

		try {
			StyleBuilder sb = new StyleBuilder();
			MapContent mc = new MapContent();

			ReferencedEnvelope mapBounds = mc.getMaxBounds();
			for (Color c : cols) {
				DefaultFeatureCollection features = new DefaultFeatureCollection();
				for (Geometry g : colMap.keySet()) {
					if (!c.equals(colMap.get(g)))
						continue;
					featureBuilder.set("the_geom", g);
					features.add(featureBuilder.buildFeature("" + features.size()));
				}
				Mark mark = sb.createMark(StyleBuilder.MARK_CIRCLE, c);
				FeatureLayer fl = new FeatureLayer(features, SLD.wrapSymbolizers(sb.createPointSymbolizer(sb.createGraphic(null, mark, null))));
				mc.addLayer(fl);
				mapBounds.expandToInclude(fl.getBounds());
			}

			GTRenderer renderer = new StreamingRenderer();
			renderer.setMapContent(mc);

			Rectangle imageBounds = null;

			double heightToWidth = mapBounds.getSpan(1) / mapBounds.getSpan(0);
			int imageWidth = 2000;
			imageBounds = new Rectangle(0, 0, imageWidth, (int) Math.round(imageWidth * heightToWidth));

			BufferedImage image = new BufferedImage(imageBounds.width, imageBounds.height, BufferedImage.TYPE_INT_ARGB);
			Graphics2D gr = image.createGraphics();
			renderer.paint(gr, imageBounds, mapBounds);

			ImageIO.write(image, "png", new FileOutputStream(fn));
			image.flush();
			mc.dispose();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
