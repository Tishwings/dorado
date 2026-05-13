Data construction.
Create a very small synthetic input BAM file of only one record.
Reason: reducing the inference time for successful tests.
  $ rm -rf data; mkdir -p data
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > in_bam=${in_dir}/in.aln.bam
  > samtools view -H ${in_bam} > data/in.micro.sam
  > samtools view ${in_bam} | head -n 1 >> data/in.micro.sam
  > samtools view -Sb data/in.micro.sam | samtools sort > data/in.micro.bam
  > samtools index data/in.micro.bam

###################################################
### Test auto-resolve for all available models  ###
### from the input BAM file.                    ###
###################################################
HAC. Auto-resolve the `dna_r10.4.1_e8.2_400bps_hac@v5.0.0_variant_mv@v1.0` model from the BAM file and use the pre-cached folder.
There should be no "downloading" log line and the process should succeed.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > in_bam="data/in.micro.bam"
  > in_ref=${in_dir}/in.ref.fasta.gz
  > model_var="--models-directory ${MODEL_ROOT_DIR}"
  > ${DORADO_BIN} smallvar --device cpu ${in_bam} ${in_ref} -t 4 --regions "chr20:1-100" -v ${model_var} > out/out.vcf 2> out/out.stderr
  > echo "Exit code: $?"
  > grep "Resolved model from input data: 'dna_r10.4.1_e8.2_400bps_hac@v5.0.0_variant_mv@v1.0'" out/out.stderr | wc -l | awk '{ print $1 }'
  > grep -- " - downloading" out/out.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  1
  0

Attempt to download a model which is not available.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > ### Create the synthetic data with no dwells.
  > samtools view -h data/in.micro.bam | sed 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.4.1_e8.2_400bps_sup@v5.0.0/g' | samtools view -Sb > out/in.modified.bam
  > samtools index out/in.modified.bam
  > ### Run the unit under test.
  > in_bam=out/in.modified.bam
  > in_ref=${in_dir}/in.ref.fasta.gz
  > ${DORADO_BIN} smallvar --device cpu ${in_bam} ${in_ref} -t 4 -v > out/out.vcf 2> out/out.stderr
  > echo "Exit code: $?"
  > grep "Resolved model from input data: 'dna_r10.4.1_e8.2_400bps_sup@v5.0.0_variant_mv@v1.0'" out/out.stderr | wc -l | awk '{ print $1 }'
  > grep -- " - downloading" out/out.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  0
  0
  [error] Could not find any variant calling model compatible with the basecaller model 'dna_r10.4.1_e8.2_400bps_sup@v5.0.0'.

##############################################
### Test auto-resolve from the Basecaller  ###
### or Polishing model name and the dwell  ###
### info in the input BAM.                 ###
##############################################
Resolve the model from a Basecaller model name `dna_r10.4.1_e8.2_400bps_hac@v5.0.0`.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > in_bam="data/in.micro.bam"
  > in_ref=${in_dir}/in.ref.fasta.gz
  > model="dna_r10.4.1_e8.2_400bps_hac@v5.0.0"
  > resolved_model="${model}_variant_mv@v1.0"
  > ${DORADO_BIN} smallvar --model-override ${model} --models-directory ${MODEL_ROOT_DIR} --device cpu ${in_bam} ${in_ref} -t 4  --regions "chr20:1-100" -v > out/out.vcf 2> out/out.stderr
  > echo "Exit code: $?"
  > grep "Resolved model from user-specified basecaller model name: '${resolved_model}'" out/out.stderr | wc -l | awk '{ print $1 }'
  > grep -- " - downloading ${resolved_model} with " out/out.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  1
  0

Resolve the model from an exact Variant Calling model name.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > in_bam="data/in.micro.bam"
  > in_ref=${in_dir}/in.ref.fasta.gz
  > model="dna_r10.4.1_e8.2_400bps_hac@v5.0.0_variant_mv@v1.0"
  > ${DORADO_BIN} smallvar --model-override ${model} --models-directory ${MODEL_ROOT_DIR} --device cpu ${in_bam} ${in_ref} -t 4 --regions "chr20:1-100" -v > out/out.vcf 2> out/out.stderr
  > echo "Exit code: $?"
  > grep "Resolved model from user-specified variant calling model name: 'dna_r10.4.1_e8.2_400bps_hac@v5.0.0_variant_mv@v1.0'" out/out.stderr | wc -l | awk '{ print $1 }'
  > grep -- " - downloading ${model} with " out/out.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  1
  0

Resolve the model from a local path.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > in_bam="data/in.micro.bam"
  > in_ref=${in_dir}/in.ref.fasta.gz
  > model=${MODEL_DIR}
  > ${DORADO_BIN} smallvar --model-override ${model} --device cpu ${in_bam} ${in_ref} -t 4 --regions "chr20:1-100" -v > out/out.vcf 2> out/out.stderr
  > echo "Exit code: $?"
  > grep "Resolved model from user-specified path: " out/out.stderr | wc -l | awk '{ print $1 }'
  > grep -- " - downloading" out/out.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.stderr | sed -E 's/.*\[/\[/g' | sed -E 's/model: .*/model/g'
  Exit code: 0
  1
  0
  [warning] Skipping basecaller compatibility checks for user-specified model

##############################################
### Compatibility checks.                  ###
##############################################
Negative test: no dwells in data, but the model uses them for polishing.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > in_bam="data/in.micro.bam"
  > ### Create the synthetic data with no dwells.
  > samtools view -H ${in_bam} > out/in.no_dwells.sam
  > samtools view ${in_bam} | cut -f1-11 >> out/in.no_dwells.sam
  > samtools view -Sb out/in.no_dwells.sam > out/in.no_dwells.bam
  > samtools index out/in.no_dwells.bam
  > ### Run the unit under test.
  > in_bam=out/in.no_dwells.bam
  > in_ref=${in_dir}/in.ref.fasta.gz
  > ${DORADO_BIN} smallvar --models-directory ${MODEL_ROOT_DIR} --device cpu ${in_bam} ${in_ref} -t 4 --infer-threads 1 -vv > out/out.vcf 2> out/out.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "Resolved model from input data: 'dna_r10.4.1_e8.2_400bps_hac@v5.0.0_variant_mv@v1.0'" out/out.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  1
  [error] Input data does not contain move tables, but a model which requires move tables has been chosen.

Passing test with warnings: Basecaller model specified in the BAM does not match the Basecaller model specified in the Variant Calling model.
Using `--model-override`.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > in_bam="data/in.micro.bam"
  > model=${MODEL_DIR}
  > ### Create the mocked data for a non-supported basecaller version (1.0.0).
  > samtools view -h ${in_bam} | sed -E 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.4.1_e8.2_400bps_hac@v1.0.0/g' | samtools view -Sb > out/in.bam
  > samtools index out/in.bam
  > ### Run the unit under test.
  > in_bam=out/in.bam
  > in_ref=${in_dir}/in.ref.fasta.gz
  > ${DORADO_BIN} smallvar --model-override ${model} --device cpu ${in_bam} ${in_ref} -t 4 --infer-threads 1 -vv > out/out.vcf 2> out/out.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.stderr | sed -E 's/.*\[/\[/g' | sed -E 's/model: .*/model/g'
  Exit code: 0
  [warning] Skipping basecaller compatibility checks for user-specified model
  [warning] Variant calling model is not compatible with the input BAM. This may produce inferior results.

Passing test with warnings: no dwells in data, but the model uses them for polishing.
Using `--model-override`.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > in_bam="data/in.micro.bam"
  > model="dna_r10.4.1_e8.2_400bps_hac@v5.0.0_variant_mv@v1.0"
  > ### Create the synthetic data with no dwells.
  > samtools view -H ${in_bam} > out/in.no_dwells.sam
  > samtools view ${in_bam} | cut -f1-11 >> out/in.no_dwells.sam
  > samtools view -Sb out/in.no_dwells.sam > out/in.no_dwells.bam
  > samtools index out/in.no_dwells.bam
  > ### Run the unit under test.
  > in_bam=out/in.no_dwells.bam
  > in_ref=${in_dir}/in.ref.fasta.gz
  > ${DORADO_BIN} smallvar --model-override "${model}" --models-directory ${MODEL_ROOT_DIR} --device cpu ${in_bam} ${in_ref} -t 4 --regions "chr20:1-100" --infer-threads 1 -vv > out/out.vcf 2> out/out.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "Resolved model from user-specified variant calling model name: 'dna_r10.4.1_e8.2_400bps_hac@v5.0.0_variant_mv@v1.0'" out/out.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  1
  [warning] Input data does not contain move tables, but a model which requires move tables has been chosen. This may produce inferior results.

Passing test with warnings: Basecaller model specified in the BAM does not match the Basecaller model specified in the Variant Calling model.
Using `--model-override`.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > in_bam="data/in.micro.bam"
  > model=${MODEL_DIR}
  > ### Create the mocked data for a non-supported basecaller version (1.0.0).
  > samtools view -h ${in_bam} | sed -E 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.4.1_e8.2_400bps_hac@v1.0.0/g' | samtools view -Sb > out/in.bam
  > samtools index out/in.bam
  > ### Run the unit under test.
  > in_bam=out/in.bam
  > in_ref=${in_dir}/in.ref.fasta.gz
  > ${DORADO_BIN} smallvar --model-override ${model} --device cpu ${in_bam} ${in_ref} -t 4 --regions "chr20:1-100" --infer-threads 1 -vv > out/out.vcf 2> out/out.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.stderr | sed -E 's/.*\[/\[/g' | sed -E 's/model: .*/model/g'
  Exit code: 0
  [warning] Skipping basecaller compatibility checks for user-specified model
  [warning] Variant calling model is not compatible with the input BAM. This may produce inferior results.

Passing test with warnings: A Polishing model is provided instead of a Variant Calling model.
Using `--model-override`.
  $ rm -rf out; mkdir -p out
  > model=${POLISH_MODEL_DIR}
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > ### Run the unit under test.
  > in_bam="data/in.micro.bam"
  > in_ref=${in_dir}/in.ref.fasta.gz
  > ${DORADO_BIN} smallvar --model-override ${model} --device cpu ${in_bam} ${in_ref} -t 4 --infer-threads 1 -vv > out/out.vcf 2> out/out.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.stderr | sed -E 's/.*\[/\[/g' | sed -E 's/model: .*/model/g'
  Exit code: 0
  [warning] Skipping basecaller compatibility checks for user-specified model
  [warning] Incompatible model label scheme! Expected DiploidLabelScheme but got HaploidLabelScheme. This may produce unexpected results.

##############################################
### Negative tests.                        ###
##############################################
Negative test: Cannot resolve the model, it does not match a Basecaller model, a Variant Calling model, or a path.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > in_bam="data/in.micro.bam"
  > model="unknown"
  > ### Run the unit under test.
  > ${DORADO_BIN} smallvar --model-override "${model}" --device cpu ${in_bam} ${in_dir}/in.ref.fasta.gz -t 4 --infer-threads 1 -vv > out/out.vcf 2> out/out.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] Could not resolve model from string: 'unknown'.

Negative test: 'auto' is not accepted as a model override alias.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > in_bam="data/in.micro.bam"
  > model="auto"
  > ### Run the unit under test.
  > ${DORADO_BIN} smallvar --model-override "${model}" --device cpu ${in_bam} ${in_dir}/in.ref.fasta.gz -t 4 --infer-threads 1 -vv > out/out.vcf 2> out/out.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] Could not resolve model from string: 'auto'.

Negative test: BAM has a model which is not available for download in auto mode.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > in_bam="data/in.micro.bam"
  > ### Create the mocked data for a non-supported basecaller version (1.0.0).
  > samtools view -h ${in_bam} | sed -E 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.4.1_e8.2_400bps_hac@v1.0.0/g' | samtools view -Sb > out/in.bam
  > samtools index out/in.bam
  > ### Run the unit under test.
  > ${DORADO_BIN} smallvar --device cpu out/in.bam ${in_dir}/in.ref.fasta.gz -t 4 --infer-threads 1 -vv > out/out.vcf 2> out/out.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] Could not find any variant calling model compatible with the basecaller model 'dna_r10.4.1_e8.2_400bps_hac@v1.0.0'.

Negative test: using auto mode but the BAM has no models listed (no RG tags).
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > in_bam="data/in.micro.bam"
  > ### Create the mocked data, remove the @RG lines.
  > samtools view -h ${in_bam} | grep -v "@RG" | samtools view -Sb > out/in.bam
  > samtools index out/in.bam
  > ### Run the unit under test.
  > ${DORADO_BIN} smallvar --device cpu out/in.bam ${in_dir}/in.ref.fasta.gz -t 4 --infer-threads 1 -vv > out/out.vcf 2> out/out.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] Input BAM file has no basecaller models listed in the header.
