bsub -P c7 -q medium -o /farmdisk1/mbonici/trained_hanyu_10000/job_loop_0_3.out -e /farmdisk1/mbonici/trained_hanyu_10000/job_loop_0_3.err -n 8 -M 4096 -R"span[hosts=1] select[mem>4096 && hname!=e4farm03 && hname!=g2farm7 && hname!=g2farm8 && hname!=teo22 && hname!=teo24 && hname!=teo25 && hname!=teo26 && hname!=teo27 && hname!=teo28 && hname!=infne01 && hname!=totem04 && hname!=totem07 && hname!=totem08 && hname!=geant15 && hname!=geant16 && hname!=aiace12 && hname!=aiace13 && hname!=aiace14 && hname!=aiace15 && hname!=aiace16 && hname!=aiace17] rusage [mem=4096]" /farmdisk1/mbonici/trained_hanyu_10000/job_loop_0_3.sh