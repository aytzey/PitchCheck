# TensorRT denemesi — 4 Ekim 2026

Hermes encoder'ın **28 decoder bloğu gerçek TensorRT motorlarına dönüştürüldü**. 42 ve 126 kelimelik sentetik girdilerde bu encoder kullanılarak gerçek TRIBE voxel tahminleri üretildi. Bu deneme canlıya alınmadı: sıcak encoder ile 126 kelimelik hesapta hız kazanımı görülmedi, sayısal eşitlik testi geçmedi ve 378 kelimelik hesap GPU bellek bütçesini aştı.

## Ortam ve dönüşüm

- Aynı paylaşılan sunucu, RTX 5080, sürücü `580.95.05`; PyTorch `2.7.1+cu128`, Transformers `5.5.0.dev0`.
- İzole imaja Torch-TensorRT `2.7.0` ve TensorRT `10.9.0.34` eklendi. Bu eşleşme [resmî 2.7 sürüm notlarında](https://github.com/pytorch/TensorRT/releases/tag/v2.7.0) belirtilir. Sistem paketleri, GPU sürücüsü ve çalışan servis imajı değiştirilmedi.
- Mevcut `NousResearch/Hermes-3-Llama-3.2-3B` ağırlıkları offline ve salt okunur mount'tan yüklendi; BF16 korundu. Quantization ve FP16 ağırlık dönüşümü uygulanmadı.
- TRIBE 29 ara çıktıyı kullandığı için encoder yalnızca son katmanı döndüren bir motorla değiştirilmedi. Mevcut Transformers'ın çıktı toplama kilidi export'u engelledi; aynı katmanları açıkça çağıran adaptör test girdisindeki 29 çıktı için native PyTorch ile **0,0 fark** verdi.
- Torch-TensorRT 2.7'nin `_to_copy` tür doğrulayıcısı BF16'yı kabul etmiyordu: 3.042 işlemin 87'si bu yüzden reddedildi. Yalnızca geçici konteynerde BF16 kabulü eklendi; aynı cast uygulaması kullanıldı. Her blok `require_full_compilation=True` ile derlendi; dönüştürülen bloklarda kalan Torch parametresi **0**.
- Tek parça encoder derlemesi 12 ve 20 GiB RAM limitlerinde konteyner OOM ile durdu. Bloklar sırayla derlenerek tam encoder oluşturuldu. Derleme sırasında konteyner RAM tavanı 12'den 16 GiB'ye çıkarıldı; canlı servisin 12 GiB sınırı korundu.
- Varsayılan blok context'leri ayrı çalışma belleği ayırınca GPU kullanımı büyüdü. Denemede TensorRT `USER_MANAGED` context'leri tek 768 MiB alanı sırayla kullandı; yardımcı stream sayısı 0, çağrılar seri. SDK'nin context/bellek API'si [IExecutionContext belgesinde](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/infer/Core/ExecutionContext.html) anlatılır. Bu deneysel paylaşım eşzamanlı çağrılar için doğrulanmadı.

## Ölçüm

İki koşuda 4 CPU, batch=4, aynı ağırlıklar, aynı sentetik metinler ve sabit feature sırası kullanıldı. Tahmin önbelleği ilk üç farklı metin için boştu. Karşılaştırma dış LLM süresini kapsamaz; `engine.score_text` süresidir.

| Girdi / ölçüm | Güncel PyTorch CUDA | TensorRT denemesi |
|---|---:|---:|
| Tek gerçek decoder bloğu; 5 sıcak ölçümün medyanı | 0,4965 ms | 0,3684 ms |
| 42 kelime; ilk skor | 0,6507 sn | 1,2549 sn |
| 126 kelime; encoder sıcak, yeni metin | 2,3397 sn | 2,4836 sn |
| 378 kelime | 15,6853 sn | GPU bütçesi aşılınca durduruldu |

Tek blokta süre %25,8 azaldı. Tam 126 kelimelik hesapta TensorRT süresi bu ölçümde **%6,15 daha uzun** oldu. Tam skorlar tek koşudur; istatistiksel hız iddiası değildir. 42 kelimelik ilk skorda TensorRT motorları GPU'ya ilk kez yüklenirken native encoder önceden GPU'da ısıtılmıştı; bu satır aynı başlangıç durumunda bir hız karşılaştırması değildir. 126 kelimelik satırda her iki encoder sıcaktı.

| Gerçek voxel tahminleri | 42 kelime | 126 kelime |
|---|---:|---:|
| Maksimum mutlak fark | 0,006168 | 0,014134 |
| Ortalama mutlak fark | 0,000652 | 0,000523 |
| Göreli L2 farkı | %0,3658 | %0,4348 |
| `rtol=1e-4, atol=1e-5` eşitlik testi | Geçmedi | Geçmedi |

Bu farklar model çıktısındaki sayısal farklardır; doğruluk veya müşteri raporu kalitesindeki değişim yüzdesi değildir. Yeni native kontrolün önceki native referansla maksimum farkı **0,0** oldu; karşılaştırmadaki fark yeni native kontrolün değişmesinden kaynaklanmadı. Tamamlanan iki TensorRT skorunda feature-cache kayıt sayısı işlem sonunda **0** idi.

Native kontrolde gözlenen süreç GPU belleği tepesi **7,342 GiB** (7.518 MiB), TensorRT denemesinde **8,174 GiB** (8.370 MiB) oldu. TensorRT 378 kelimelik metnin feature çıkarımında 8 GiB sınırını aşınca gözlemci yalnızca deneme konteynerini öldürdü; bu CUDA OOM değildir. Gözlemci 1 saniyelik örneklerle takip eder, donanımsal sert kota oluşturmaz. Ayrıca hostta 12 GiB kullanılabilir RAM ve GPU'da 2 GiB boş bellek alt sınırları vardı. Son tamamlanan TensorRT skoru sonrası kaydedilen süreç MaxRSS yaklaşık **13,91 GiB**; derleme ve yükleme de bu kayda dahildir.

## Canlı servis ve karar

Canlı `pitchserver_tribe` imajı aynı kaldı: `sha256:9572265cf67a3be3186ff479552126aa370f06deb93240a7563470d010e50721`. Sağlık durumu `healthy`, restart sayısı 1. Başlangıç/son karşılaştırmasında **166 konteynerin tamamı**, dolayısıyla diğer **165 servis**, aynı kimlik/başlangıç/restart/sağlık kayıtlarını korudu. Deneme konteynerleri kaldırıldı. Son GPU kullanımı 4.291 MiB'ye döndü.

Bu aday hız, çıktı eşitliği ve kaynak bütçesi kontrollerini birlikte geçmedi. Canlı backend güncel PyTorch CUDA sürümünü kullanır. TensorRT'nin bütün uygulamalar veya başka derleme seçenekleri için daha yavaş olduğu sonucu çıkarılmaz.

## Tekrar üretme

Sunucudaki kanıt dizini:

```text
/home/dkmserver/Desktop/Machinity/aytug/pitchserver/audit-tensorrt-20261004/
```

`comparison.json` son karşılaştırma; `layerwise/` tam encoder denemesinin script/log/sonuçları; `block/` tek blok ölçümü; `native-warm/` native kontrol ve `.npy` matrisleri; `service-verification.json` servis karşılaştırmasıdır. Önceki denemeler `attempt-*` altında tutulur. `layerwise/probe-result.json` içindeki `compiled:true` motorların oluştuğunu belirtir; son kabul veya bütün girdilerin tamamlandığı anlamına gelmez. GPU gözlemcisinin durdurduğu işi `layerwise/container-result.json` kanıtlar.

İmaj `pitchcheck-tensorrt-experiment:20261004` etiketiyle sunucudadır. Yeniden oluşturmak için doğrulanan native imajı `pitchcheck-trt-base:20261004` olarak etiketleyip [Dockerfile.tensorrt-experiment](../tribe_service/Dockerfile.tensorrt-experiment) kullanılır. Sürücü ve tüm GPU bağımlılıkları tekrar kurulmaz.

[experiment-tensorrt.py](../scripts/experiment-tensorrt.py) aynı dönüşüm yöntemini içerir. Yalnızca geçici Docker konteynerinde `PITCHCHECK_TRT_EXPERIMENT=1` ile çalışır; test sırasında o konteynerin TensorRT Python dosyalarını değiştirir. `/audit` yazılabilir deney dizini, `/baseline` native referans `.npy` dosyaları, `/weights` mevcut model dizininin salt okunur mount'u olmalıdır. `HF_HOME=/weights/huggingface`, offline mod, `TRIBE_CACHE_DIR=/audit/features-layerwise`, CUDA, batch=4 ve OMP/MKL/OpenBLAS=2 kullanılır. Üretim auth/env dosyaları mount edilmez; port açılmaz.

`run-tensorrt.py` / `run-native.py` CPU, RAM, PID, GPU ve boş bellek takibini uygular. TensorRT scripti için 16 GiB, native kontrol için 12 GiB RAM tavanı kullanılır. Başarılı native kontrol 42/126/378 kelimelik girdileri ve önbellek tekrarını tamamladı. Denemeyi gözlemci olmadan veya canlı servisin içinde çalıştırmayın.

Kontrollü tekrar için, diğer GPU işleri ve boş RAM kontrol edildikten sonra:

```bash
ssh -p 2022 -o IdentityAgent=none dkmserver@78.186.120.189
cd /home/dkmserver/Desktop/Machinity/aytug/pitchserver/audit-tensorrt-20261004
python3 run-tensorrt.py
```

Bu komut yalnızca geçici deney konteynerini oluşturur/kaldırır; canlı servisi yeniden başlatmaz. GPU bütçesi veya diğer durdurma koşulları tekrar devreye girebilir. Çıktılar son adayın üretime uygun olduğunu göstermez; script `eligible_for_deployment:false` kaydeder.
