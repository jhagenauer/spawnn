package llm.ga;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.apache.log4j.Logger;

import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryCollection;
import com.vividsolutions.jts.geom.GeometryFactory;

import ga.GeneticAlgorithm;
import llm.LLMNG;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.utils.NGUtils;
import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;
import spawnn.utils.DataUtils.Transform;

public class LLM_GA_Main {

	private static Logger log = Logger.getLogger(LLM_GA_Main.class);

	public static void main(String[] args) {
		
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/lucas/lucas.shp"), true);
		int[] fa = new int[] { 
				2, // TLA
				3, // beds
				9, // rooms
				10, // lotsize
				19, // age
		};
		int[] ga = new int[] { 20, 21 };
		int ta = 0;

		for (double[] d : sdf.samples) {
			d[19] = Math.pow(d[19], 2);
			d[10] = Math.log(d[10]);
			d[2] = Math.log(d[2]);
			d[1] = Math.log(d[1]);
		}
		
		LLM_CostCalculator cc = new LLM_CostCalculator(sdf.samples,ga,fa,ta);
		
		LLM_Individual[] goodIndividuals = new LLM_Individual[] {
		};
		
		GeneticAlgorithm.tournamentSize = 3;
		GeneticAlgorithm.elitist = true;
		GeneticAlgorithm.recombProb = 0.7;
		
		for( LLM_Individual i : goodIndividuals ) {
			log.debug(cc.getCost(i));
			
			Sorter<double[]> sorter = i.getSorter(ga, fa);
			
			List<double[]> nSamples = new ArrayList<>();
			for( double[] d : sdf.samples )
				nSamples.add( Arrays.copyOf(d, d.length));
			DataUtils.transform(nSamples, Transform.zScore);
			
			LLMNG llmng = i.buildModel(nSamples, ga, fa, ta);
			Map<double[],Set<double[]>> mapping = NGUtils.getBmuMapping(nSamples, llmng.getNeurons(), sorter);
			
			// save llmng neuron
			GeometryFactory gf = new GeometryFactory();
			List<Geometry> geoms = new ArrayList<>();
			List<double[]> smpls = new ArrayList<>();
			for( Entry<double[], Set<double[]>> e : mapping.entrySet() ) {
				
				Geometry[] gs = new Geometry[e.getValue().size()];
				int j = 0;
				for( double[] d : e.getValue() ) {
					int idx = nSamples.indexOf(d);
					gs[j++] = sdf.geoms.get(idx);
				}
				GeometryCollection gc = new GeometryCollection( gs, gf );
				geoms.add( gc.union() );
				
				double[] neuron = e.getKey();
				double[] output = llmng.output.get(neuron);
				double[] matrix = llmng.matrix.get(neuron)[0];
								
				double[] d = new double[fa.length*2+1];
				for( int k = 0; k < fa.length; k++ ) {
					d[k] = neuron[fa[k]];
					d[k+fa.length] = matrix[fa[k]];
				}	
				d[2*fa.length] = output[0];
				smpls.add(d);
			}
			
			String[] names = new String[fa.length*2+1];
			for( int k = 0; k < fa.length; k++ ) {
				names[k] = sdf.getNames()[fa[k]];
				names[k+fa.length] = "m_"+sdf.getNames()[fa[k]];
				}
			names[2*fa.length] = "output";
			
			DataUtils.writeShape(smpls, geoms, names, sdf.crs, "output/"+i.hashCode()+".shp");
		}
		System.exit(1);
		
		List<LLM_Individual> init = new ArrayList<>();
		while (init.size() < 100)
			init.add(new LLM_Individual());

		GeneticAlgorithm<LLM_Individual> gen = new GeneticAlgorithm<LLM_Individual>();
		LLM_Individual result = (LLM_Individual) gen.search(init, cc);

		log.info("best:");
		log.info(cc.getCost(result));
		log.info(result.iParam);
	}
}
