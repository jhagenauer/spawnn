package spawnn.gui;

import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.util.List;

import javax.swing.JFrame;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JTabbedPane;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import net.miginfocom.swing.MigLayout;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.utils.SpatialDataFrame;

public class SpawnnGui extends JFrame implements PropertyChangeListener, ActionListener, ChangeListener, TrainingListener { //TODO A general class Prototype for GridPos, double[] etc. is needed

	private static final long serialVersionUID = -2728973134956990131L;

	private DataPanel dataPanel;
	private AnnPanel annPanel;

	private JTabbedPane tp;
	private JMenuItem aboutItem, quitItem;

	SpawnnGui() {
		super("Spawnn");

		// Menu Bar
		JMenuBar mb = new JMenuBar();
		mb.setLayout(new MigLayout("insets 0"));
		setJMenuBar(mb);

		JMenu fileMenu = new JMenu("File");
		
		quitItem = new JMenuItem("Quit");
		quitItem.addActionListener(this);
		fileMenu.add(quitItem);

		JMenu helpMenu = new JMenu("Help");
		aboutItem = new JMenuItem("About");
		aboutItem.addActionListener(this);
		helpMenu.add(aboutItem);

		mb.add(fileMenu, "push");
		mb.add(helpMenu);

		// Tabs
		tp = new JTabbedPane();
		dataPanel = new DataPanel();
		dataPanel.addPropertyChangeListener(this);

		tp.addTab("Data", dataPanel);

		annPanel = new AnnPanel(this);
		annPanel.addPropertyChangeListener(this);
		annPanel.addTrainingListener(this);

		tp.addTab("ANN", annPanel);
		tp.setEnabledAt(1, false);

		add(tp);
		tp.addChangeListener(this);

		pack();
		setSize(1245, 700);
		setMinimumSize(new Dimension(1245, 700)); // ugly, but needed because of ng results minimum size (vv)

		setVisible(true);
	}

	public static void main(String[] args) {
		/*try {
			UIManager.setLookAndFeel("javax.swing.plaf.nimbus.NimbusLookAndFeel");
		} catch (Exception e) {
			try {
				UIManager.setLookAndFeel(UIManager.getCrossPlatformLookAndFeelClassName());
			} catch (Exception ex) {
				ex.printStackTrace();
			}
		}*/
		new SpawnnGui();
	}

	@Override
	public void propertyChange(PropertyChangeEvent e) {
		if (e.getPropertyName().equals(DataPanel.TRAIN_ALLOWED_PROP) )
			tp.setEnabledAt(1, (Boolean)e.getNewValue() );
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		if (e.getSource() == aboutItem) {
			JOptionPane.showMessageDialog(this, "Spatial Analysis with (Self-Organinzing) Neural Networks (SPAWNN) ©2013-2015\n\n" + "All rights reserved.\n\n" + "Julian Hagenauer", "About", JOptionPane.INFORMATION_MESSAGE);
		} else if (e.getSource() == quitItem) {
			System.exit(1);
		} 
	}
	
	@Override
	public void stateChanged(ChangeEvent ce) {
		if( ((JTabbedPane)ce.getSource()).getSelectedComponent() == annPanel ) { // submit data for training
			int[] gaUsed = dataPanel.getGA(false);
			int[] gaAll = dataPanel.getGA(true);
			annPanel.setTrainingData(dataPanel.getNormedSamples(), dataPanel.getSpatialData(), dataPanel.getFA(), gaUsed, gaAll);
		}
	}

	int numSom = 0;
	int numNg = 0;

	@Override
	public void trainingResultsAvailable(TrainingEvent te) {
		int[] fa = dataPanel.getFA();
		Dist<double[]> fDist = new EuclideanDist(fa);
		Dist<double[]> gDist = new EuclideanDist(dataPanel.getGA(false));
		List<double[]> samples = te.getSamples();
		SpatialDataFrame sdf = dataPanel.getSpatialData();
		int[] ga = dataPanel.getGA(true);
		
		if( te instanceof TrainingFinishedSOM ) { // SOM-results
			TrainingFinishedSOM tfs = (TrainingFinishedSOM)te;
			SOMResultPanel srp = new SOMResultPanel(this, sdf, samples, tfs.getBmus(), tfs.getGrid(), fDist, gDist, fa, ga, tfs.isWMC());
			tp.addTab("Results (SOM," + (numSom++) + ")", srp);
			tp.setSelectedComponent(srp);
			tp.setTabComponentAt(tp.indexOfComponent(srp), new ButtonTabComponent(tp));
		} else {
			TrainingFinishedNG tfn = (TrainingFinishedNG)te;
			NGResultPanel srp = new NGResultPanel(this, sdf, samples, tfn.getBmus(), tfn.getGraph(), fDist, gDist, fa, ga, tfn.isWMC() );
			tp.addTab("Results (NG, " + (numNg++) + ")", srp);
			tp.setSelectedComponent(srp);
			tp.setTabComponentAt(tp.indexOfComponent(srp), new ButtonTabComponent(tp));
		}
	}
}
